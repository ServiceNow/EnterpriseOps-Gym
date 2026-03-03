import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    Uses LangChain for consistent tool calling interface.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        api_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        region: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
        effort: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.custom_api_endpoint = api_endpoint
        self.custom_api_version = api_version
        self.region = region
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.effort = effort
        self.llm = None

        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        try:
            if self.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                self.llm = ChatAnthropic(
                    model=self.model,
                    anthropic_api_key=self.api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            elif self.provider == "aws_bedrock":
                from langchain_aws import ChatBedrock

                self.llm = ChatBedrock(
                    model_id=self.model,
                    region_name=self.region or "us-west-2",
                    model_kwargs={
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                )
            elif self.provider == "openai":
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    model=self.model,
                    openai_api_key=self.api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            elif self.provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            elif self.provider == "googlevertexai":
                from langchain_google_vertexai import ChatVertexAI

                self.llm = ChatVertexAI(
                    model=self.model,
                    project=self.api_key,
                    location=self.region or "global",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            # MARKER: Add a SN provider here
            elif self.provider == "azureopenai":
                from langchain_openai import AzureChatOpenAI

                model_kwargs = {}
                if self.top_p is not None:
                    model_kwargs["top_p"] = self.top_p
                if self.effort is not None:
                    model_kwargs["reasoning_effort"] = self.effort

                self.llm = AzureChatOpenAI(
                    azure_endpoint=self.custom_api_endpoint,
                    api_key=self.api_key,
                    api_version=self.custom_api_version,
                    azure_deployment=self.model,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    model_kwargs=model_kwargs # In GPT-5.X this is a first class parameter, but passing this way is also allowed.
                )
            elif self.provider == "vllm" or self.provider == "openrouter":
                from langchain_openai import ChatOpenAI

                model_kwargs = {}
                if self.top_p is not None:
                    model_kwargs["top_p"] = self.top_p
                if self.effort is not None:
                    model_kwargs["reasoning_effort"] = self.effort
                # vLLM exposes an OpenAI-compatible API
                self.llm = ChatOpenAI(
                    model=self.model,
                    openai_api_key=self.api_key or "not-needed",
                    openai_api_base=self.custom_api_endpoint,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    model_kwargs=model_kwargs,
                )
            elif self.provider == "qwq":
                from langchain_qwq import ChatQwQ

                self.llm = ChatQwQ(
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.custom_api_endpoint,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **({"top_p": self.top_p} if self.top_p is not None else {}),
                )
            elif self.provider == "deepseek":
                from langchain_deepseek import ChatDeepSeek

                # Default to "medium" if effort is not provided or empty
                effort_value = self.effort if self.effort else "medium"

                self.llm = ChatDeepSeek(
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.custom_api_endpoint,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    model_kwargs={
                        "extra_body": {
                            "reasoning": {
                                "enabled": True,
                                "effort": effort_value
                            }
                        }
                    },
                    **({"top_p": self.top_p} if self.top_p is not None else {}),
                )

            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")

            logger.info(f"Initialized {self.provider} LLM: {self.model}")

        except ImportError as e:
            logger.error(
                f"Failed to import LangChain provider for {self.provider}: {e}"
            )
            raise

    def _convert_mcp_tools_to_langchain(
        self, mcp_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert MCP tool definitions to LangChain format, filtering incompatible schemas"""
        langchain_tools = []

        for tool in mcp_tools:
            input_schema = tool.get(
                "inputSchema", {"type": "object", "properties": {}, "required": []}
            )

            # Clean schema: remove oneOf, allOf, anyOf at top level (Anthropic doesn't support them)
            cleaned_schema = self._clean_json_schema(input_schema)

            tool_def = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": cleaned_schema,
                },
            }
            langchain_tools.append(tool_def)

        return langchain_tools

    def _clean_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean JSON schema to be compatible with LLM APIs (Anthropic, Vertex AI, etc.)

        Removes oneOf, allOf, anyOf at top level and converts to simple object schema.
        Also handles optional parameters with type arrays like ['STRING', 'NULL'].
        """
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "required": []}

        # If schema has oneOf/allOf/anyOf at top level, extract the first valid object schema
        if "oneOf" in schema:
            logger.debug(
                f"Schema has oneOf at top level, extracting first object schema"
            )
            for option in schema["oneOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                # No object schema found, return empty
                return {"type": "object", "properties": {}, "required": []}

        if "allOf" in schema:
            logger.debug(f"Schema has allOf at top level, merging schemas")
            merged_schema = {"type": "object", "properties": {}, "required": []}
            for sub_schema in schema["allOf"]:
                if isinstance(sub_schema, dict):
                    if "properties" in sub_schema:
                        merged_schema["properties"].update(sub_schema["properties"])
                    if "required" in sub_schema:
                        merged_schema["required"].extend(sub_schema["required"])
            schema = merged_schema

        if "anyOf" in schema:
            logger.debug(
                f"Schema has anyOf at top level, extracting first object schema"
            )
            for option in schema["anyOf"]:
                if isinstance(option, dict) and option.get("type") == "object":
                    schema = option
                    break
            else:
                return {"type": "object", "properties": {}, "required": []}

        # Ensure schema has required fields
        if "type" not in schema:
            schema["type"] = "object"

        if schema["type"] == "object" and "properties" not in schema:
            schema["properties"] = {}

        # Clean property schemas recursively to handle optional parameters
        # Vertex AI doesn't support type arrays like ['STRING', 'NULL']
        if "properties" in schema:
            cleaned_properties = {}
            optional_properties = []  # Track properties that should not be required

            for prop_name, prop_schema in schema["properties"].items():
                if isinstance(prop_schema, dict):
                    cleaned_prop = prop_schema.copy()

                    # Handle type arrays like ['STRING', 'NULL'] or ['string', 'null']
                    if "type" in cleaned_prop and isinstance(cleaned_prop["type"], list):
                        type_list = cleaned_prop["type"]
                        # Filter out 'null' or 'NULL' and take the first non-null type
                        non_null_types = [t for t in type_list if t.lower() != "null"]
                        if non_null_types:
                            cleaned_prop["type"] = non_null_types[0]
                            # If NULL was in the list, this is an optional parameter
                            if len(non_null_types) < len(type_list):
                                optional_properties.append(prop_name)
                                logger.debug(f"Property '{prop_name}' has optional type {type_list}, converted to {non_null_types[0]}")
                        else:
                            # All types were null, default to string
                            cleaned_prop["type"] = "string"
                            optional_properties.append(prop_name)

                    # Recursively clean nested object schemas
                    if cleaned_prop.get("type") == "object" and "properties" in cleaned_prop:
                        cleaned_prop = self._clean_json_schema(cleaned_prop)

                    # Handle arrays with item schemas
                    if cleaned_prop.get("type") == "array" and "items" in cleaned_prop:
                        if isinstance(cleaned_prop["items"], dict):
                            cleaned_prop["items"] = self._clean_json_schema(cleaned_prop["items"])

                    cleaned_properties[prop_name] = cleaned_prop
                else:
                    cleaned_properties[prop_name] = prop_schema

            schema["properties"] = cleaned_properties

            # Remove optional properties from the required array
            if "required" in schema and optional_properties:
                schema["required"] = [
                    req for req in schema["required"]
                    if req not in optional_properties
                ]
                if optional_properties:
                    logger.debug(f"Removed optional properties from required list: {optional_properties}")

        return schema

    async def invoke_with_tools(
        self, messages: List[Any], tools: List[Dict[str, Any]]
    ) -> Any:
        """Invoke LLM with tools"""
        # Convert MCP tools to LangChain format
        langchain_tools = self._convert_mcp_tools_to_langchain(tools)

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(langchain_tools)
        llm_with_retry = llm_with_tools.with_retry(
            retry_if_exception_type=(
                Exception,  # MARKER: You can be more specific, but for now catch all and retry.
            ),
            wait_exponential_jitter=True,
            stop_after_attempt=3,  # Retry up to 3 times
        )
        # Invoke
        logger.info(f"Invoking {self.provider} LLM with {len(tools)} tools")
        response = await llm_with_retry.ainvoke(messages)
        return response
