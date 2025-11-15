import asyncio
import io
import json
import re
import base64
from typing import Callable, Any, Coroutine, Optional
from json_repair import repair_json

from anthropic import AsyncAnthropic
from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
    ImageBlockParam,
    ToolUseBlock,
    ToolResultBlockParam,
    MessageStreamEvent,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    MessageStopEvent,
    MessageDeltaEvent,
)

from src.config.api_ada_configs import ModelInfo, APIProvider
from src.common.logger import get_logger
from .base_client import APIResponse, UsageRecord, BaseClient, client_registry
from ..exceptions import (
    RespParseException,
    NetworkConnectionError,
    RespNotOkException,
    ReqAbortException,
    EmptyResponseException,
)
from ..payload_content.message import Message as InternalMessage, RoleType
from ..payload_content.resp_format import RespFormat
from ..payload_content.tool_option import ToolOption, ToolParam, ToolCall

logger = get_logger("Claude客户端")


def _convert_messages(messages: list[InternalMessage]) -> tuple[list[MessageParam], Optional[str]]:
    """
    转换消息格式 - 将消息转换为Anthropic API所需的格式
    :param messages: 消息列表
    :return: (转换后的消息列表, system消息字符串或None)
    """
    converted = []
    system_message_parts = []
    
    for message in messages:
        # Skip system messages - they'll be returned separately
        if message.role == RoleType.System:
            if isinstance(message.content, str):
                system_message_parts.append(message.content)
            elif isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, str):
                        system_message_parts.append(item)
            continue
        
        # Handle tool results (for tool role)
        if message.role == RoleType.Tool and message.tool_call_id:
            # Anthropic uses tool_use_id instead of tool_call_id
            tool_content = message.content if isinstance(message.content, str) else str(message.content)
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": tool_content
                }]
            })
            continue
        
        # Determine role for user/assistant messages
        if message.role == RoleType.User:
            role = "user"
        elif message.role == RoleType.Assistant:
            role = "assistant"
        else:
            # Skip unknown roles
            continue
        
        # Handle content
        content: list[TextBlock | ImageBlockParam] = []
        
        if isinstance(message.content, str):
            content.append({"type": "text", "text": message.content})
        elif isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, tuple):
                    # Image content: (format, base64_data)
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{item[0].lower()}",
                            "data": item[1]
                        }
                    })
                elif isinstance(item, str):
                    content.append({"type": "text", "text": item})
        
        if content:  # Only add if content is not empty
            converted.append({
                "role": role,
                "content": content
            })
    
    system_message = "\n".join(system_message_parts) if system_message_parts else None
    return converted, system_message  # type: ignore


def _convert_tool_options(tool_options: list[ToolOption]) -> list[dict[str, Any]]:
    """
    转换工具选项格式 - 将工具选项转换为Anthropic API所需的格式
    :param tool_options: 工具选项列表
    :return: 转换后的工具选项列表
    """
    tools = []
    
    for tool_option in tool_options:
        tool_def: dict[str, Any] = {
            "name": tool_option.name,
            "description": tool_option.description or "",
            "input_schema": {}
        }
        
        # Convert parameters to JSON schema
        if tool_option.parameters:
            properties = {}
            required = []
            
            for param in tool_option.parameters:
                param_schema: dict[str, Any] = {
                    "type": param.type.value if hasattr(param.type, 'value') else str(param.type),
                    "description": param.description or ""
                }
                
                if param.enum:
                    param_schema["enum"] = param.enum
                
                properties[param.name] = param_schema
                
                if param.required:
                    required.append(param.name)
            
            tool_def["input_schema"] = {
                "type": "object",
                "properties": properties,
                "required": required if required else None
            }
            # Remove required if empty
            if not tool_def["input_schema"]["required"]:
                del tool_def["input_schema"]["required"]
        
        tools.append(tool_def)
    
    return tools


def _extract_tool_calls_from_content(content: list[Any]) -> list[ToolCall]:
    """
    从Anthropic响应内容中提取工具调用
    :param content: Anthropic响应内容列表
    :return: ToolCall列表
    """
    tool_calls = []
    
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_use_id = block.get("id", "")
            function_name = block.get("name", "")
            arguments = block.get("input", {})
            
            tool_calls.append(ToolCall(tool_use_id, function_name, arguments))
        elif hasattr(block, "type") and block.type == "tool_use":
            # Handle Anthropic SDK objects
            tool_use_id = getattr(block, "id", "")
            function_name = getattr(block, "name", "")
            arguments = getattr(block, "input", {})
            
            tool_calls.append(ToolCall(tool_use_id, function_name, arguments))
    
    return tool_calls


async def _default_stream_response_handler(
    resp_stream: Any,
    interrupt_flag: asyncio.Event | None,
) -> tuple[APIResponse, Optional[tuple[int, int, int]]]:
    """
    流式响应处理函数 - 处理Anthropic API的流式响应
    :param resp_stream: 流式响应对象
    :param interrupt_flag: 中断信号量
    :return: (APIResponse对象, 使用情况记录)
    """
    _content_buffer = io.StringIO()
    _tool_calls_buffer: list[tuple[str, str, io.StringIO]] = []
    _usage_record: Optional[tuple[int, int, int]] = None
    _current_tool_call: Optional[tuple[str, str, io.StringIO]] = None
    
    try:
        async for event in resp_stream:
            if interrupt_flag and interrupt_flag.is_set():
                raise ReqAbortException("请求被外部信号中断")
            
            if isinstance(event, ContentBlockStartEvent):
                if event.content_block.type == "tool_use":
                    tool_use = event.content_block
                    _current_tool_call = (
                        tool_use.id,
                        tool_use.name,
                        io.StringIO()
                    )
                    _tool_calls_buffer.append(_current_tool_call)
            
            elif isinstance(event, ContentBlockDeltaEvent):
                if event.delta.type == "text_delta":
                    _content_buffer.write(event.delta.text)
                elif event.delta.type == "tool_use_delta" and _current_tool_call:
                    if hasattr(event.delta, "partial_json"):
                        _current_tool_call[2].write(event.delta.partial_json)
            
            elif isinstance(event, MessageDeltaEvent):
                if hasattr(event, "usage"):
                    usage = event.usage
                    _usage_record = (
                        getattr(usage, "input_tokens", 0) or 0,
                        getattr(usage, "output_tokens", 0) or 0,
                        (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
                    )
            
            elif isinstance(event, MessageStopEvent):
                if hasattr(event, "message") and hasattr(event.message, "usage"):
                    usage = event.message.usage
                    _usage_record = (
                        getattr(usage, "input_tokens", 0) or 0,
                        getattr(usage, "output_tokens", 0) or 0,
                        (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
                    )
        
        # Build response
        resp = APIResponse()
        resp.content = _content_buffer.getvalue()
        _content_buffer.close()
        
        # Parse tool calls
        if _tool_calls_buffer:
            resp.tool_calls = []
            for call_id, function_name, arguments_buffer in _tool_calls_buffer:
                if arguments_buffer.tell() > 0:
                    raw_arg_data = arguments_buffer.getvalue()
                    arguments_buffer.close()
                    try:
                        arguments = json.loads(repair_json(raw_arg_data))
                        if not isinstance(arguments, dict):
                            raise RespParseException(
                                None,
                                f"响应解析失败，工具调用参数无法解析为字典类型。工具调用参数原始响应：\n{raw_arg_data}",
                            )
                    except json.JSONDecodeError as e:
                        raise RespParseException(
                            None,
                            f"响应解析失败，无法解析工具调用参数。工具调用参数原始响应：{raw_arg_data}",
                        ) from e
                else:
                    arguments_buffer.close()
                    arguments = {}
                
                resp.tool_calls.append(ToolCall(call_id, function_name, arguments))
        
        if not resp.content and not resp.tool_calls:
            raise EmptyResponseException()
        
        return resp, _usage_record
    
    except Exception as e:
        _content_buffer.close()
        for _, _, buffer in _tool_calls_buffer:
            buffer.close()
        raise


def _default_normal_response_parser(
    resp: Message,
) -> tuple[APIResponse, Optional[tuple[int, int, int]]]:
    """
    解析对话响应 - 将Anthropic API响应解析为APIResponse对象
    :param resp: 响应对象
    :return: (APIResponse对象, 使用情况记录)
    """
    api_response = APIResponse()
    
    # Extract text content
    text_parts = []
    for block in resp.content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        elif hasattr(block, "type") and block.type == "text":
            text_parts.append(getattr(block, "text", ""))
    
    api_response.content = "".join(text_parts)
    
    # Extract tool calls
    api_response.tool_calls = _extract_tool_calls_from_content(resp.content)
    
    # Extract usage
    usage_record = None
    if hasattr(resp, "usage") and resp.usage:
        usage = resp.usage
        usage_record = (
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
        )
    
    if not api_response.content and not api_response.tool_calls:
        raise EmptyResponseException()
    
    return api_response, usage_record


@client_registry.register_client_class("claude")
class ClaudeClient(BaseClient):
    def __init__(self, api_provider: APIProvider):
        super().__init__(api_provider)
        self.client = AsyncAnthropic(
            api_key=api_provider.api_key,
            base_url=api_provider.base_url,
            timeout=api_provider.timeout,
        )
    
    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[InternalMessage],
        tool_options: list[ToolOption] | None = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.7,
        response_format: RespFormat | None = None,
        stream_response_handler: Optional[
            Callable[[Any, asyncio.Event | None], Coroutine[Any, Any, tuple[APIResponse, Optional[tuple[int, int, int]]]]]
        ] = None,
        async_response_parser: Optional[
            Callable[[Any], tuple[APIResponse, Optional[tuple[int, int, int]]]]
        ] = None,
        interrupt_flag: asyncio.Event | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取对话响应
        Args:
            model_info: 模型信息
            message_list: 对话体
            tool_options: 工具选项（可选，默认为None）
            max_tokens: 最大token数（可选，默认为1024）
            temperature: 温度（可选，默认为0.7）
            response_format: 响应格式（可选，默认为None，Anthropic不支持此参数）
            stream_response_handler: 流式响应处理函数（可选）
            async_response_parser: 响应解析函数（可选）
            interrupt_flag: 中断信号量（可选，默认为None）
            extra_params: 额外参数（可选）
        Returns:
            APIResponse对象
        """
        if stream_response_handler is None:
            stream_response_handler = _default_stream_response_handler
        
        if async_response_parser is None:
            async_response_parser = _default_normal_response_parser
        
        # Convert messages to Anthropic format
        messages, system_message = _convert_messages(message_list)
        
        # Validate messages - Anthropic requires at least one message
        if not messages:
            raise RespParseException(
                None,
                "Anthropic API requires at least one message in the messages array"
            )
        
        # Convert tools
        tools = _convert_tool_options(tool_options) if tool_options else None
        
        # Prepare request parameters
        request_params: dict[str, Any] = {
            "model": model_info.model_identifier,
            "messages": messages,
            "max_tokens": max_tokens or 1024,
        }
        
        # Temperature is optional for Anthropic
        if temperature is not None:
            request_params["temperature"] = temperature
        
        if system_message:
            request_params["system"] = system_message
        
        if tools:
            request_params["tools"] = tools
        
        if extra_params:
            request_params.update(extra_params)
        
        # Log request for debugging (without sensitive data)
        logger.debug(f"Claude API request: model={model_info.model_identifier}, messages_count={len(messages)}, has_system={bool(system_message)}, has_tools={bool(tools)}")
        
        try:
            if model_info.force_stream_mode:
                # For streaming, create the stream directly
                if interrupt_flag and interrupt_flag.is_set():
                    raise ReqAbortException("请求被外部信号中断")
                
                stream = await self.client.messages.create(**request_params, stream=True)
                resp, usage_record = await stream_response_handler(stream, interrupt_flag)
            else:
                # For non-streaming, await directly
                if interrupt_flag and interrupt_flag.is_set():
                    raise ReqAbortException("请求被外部信号中断")
                
                raw_response = await self.client.messages.create(**request_params, stream=False)
                # async_response_parser is a sync function, don't await it
                resp, usage_record = async_response_parser(raw_response)
        
        except Exception as e:
            error_msg = str(e)
            status_code = 500
            
            # Check for Anthropic-specific error types
            if hasattr(e, "status_code"):
                status_code = getattr(e, "status_code", 500)
            elif hasattr(e, "response"):
                # Try to get status code from response
                if hasattr(e.response, "status_code"):
                    status_code = e.response.status_code
            
            # Get more detailed error message if available
            error_detail = error_msg
            if hasattr(e, "response"):
                try:
                    if hasattr(e.response, "text"):
                        error_detail = e.response.text
                    elif hasattr(e.response, "json"):
                        error_json = e.response.json()
                        if isinstance(error_json, dict):
                            error_detail = error_json.get("error", {}).get("message", error_msg)
                except Exception:
                    pass
            
            logger.error(f"Claude API error (status {status_code}): {error_detail}", exc_info=True)
            
            if "connection" in error_msg.lower() or "network" in error_msg.lower() or "timeout" in error_msg.lower():
                raise NetworkConnectionError() from e
            else:
                raise RespNotOkException(status_code, error_detail) from e
        
        if usage_record:
            resp.usage = UsageRecord(
                model_name=model_info.name,
                provider_name=model_info.api_provider,
                prompt_tokens=usage_record[0],
                completion_tokens=usage_record[1],
                total_tokens=usage_record[2],
            )
        
        return resp
    
    async def get_embedding(
        self,
        model_info: ModelInfo,
        embedding_input: str,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取文本嵌入
        Note: Anthropic doesn't provide embedding models, this will raise an error
        :param model_info: 模型信息
        :param embedding_input: 嵌入输入文本
        :param extra_params: 额外参数
        :return: 嵌入响应
        """
        raise NotImplementedError("Anthropic API does not support embeddings")
    
    async def get_audio_transcriptions(
        self,
        model_info: ModelInfo,
        audio_base64: str,
        max_tokens: Optional[int] = None,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取音频转录
        Note: Anthropic doesn't provide audio transcription, this will raise an error
        :param model_info: 模型信息
        :param audio_base64: base64编码的音频数据
        :param max_tokens: 最大token数
        :param extra_params: 额外参数
        :return: 音频转录响应
        """
        raise NotImplementedError("Anthropic API does not support audio transcription")
    
    def get_support_image_formats(self) -> list[str]:
        """
        获取支持的图片格式
        :return: 支持的图片格式列表
        """
        return ["jpg", "jpeg", "png", "webp", "gif"]

