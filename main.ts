// Import required modules
import { Application, Router, Context, Request } from "https://deno.land/x/oak@v12.6.1/mod.ts";
import { config } from "https://deno.land/x/dotenv@v3.2.2/mod.ts";

// Load environment variables
const env = config();

// Configure logging
const logger = {
  info: (message: string) => console.log(`INFO: ${new Date().toISOString()} - ${message}`),
  warning: (message: string) => console.warn(`WARNING: ${new Date().toISOString()} - ${message}`),
  error: (message: string) => console.error(`ERROR: ${new Date().toISOString()} - ${message}`),
  exception: (message: string) => console.error(`EXCEPTION: ${new Date().toISOString()} - ${message}`),
  debug: (message: string) => console.log(`DEBUG: ${new Date().toISOString()} - ${message}`)
};

// Create Oak application
const app = new Application();

// CORS middleware
app.use(async (ctx, next) => {
  ctx.response.headers.set("Access-Control-Allow-Origin", "*");
  ctx.response.headers.set("Access-Control-Allow-Methods", "*");
  ctx.response.headers.set("Access-Control-Allow-Headers", "*");
  ctx.response.headers.set("Access-Control-Allow-Credentials", "true");
  
  if (ctx.request.method === "OPTIONS") {
    ctx.response.status = 204;
    return;
  }
  
  await next();
});

// Configuration
const DEEPSIDER_API_BASE = "https://api.chargpt.ai/api/v2";
let TOKEN_INDEX = 0;

// Model mapping table
const MODEL_MAPPING: Record<string, string> = {
  "gpt-4o-mini": "openai/gpt-4o-mini",
  "gpt-4o": "openai/gpt-4o",
  "o1": "openai/o1",
  "o3-mini": "oopenai/o3-mini",
  "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
  "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
  "grok-3": "x-ai/grok-3",
  "grok-3-reasoner":"x-ai/grok-3-reasoner",
  "deepseek-v3":"deepseek/deepseek-chat",
  "deepseek-r1":"deepseek/deepseek-r1",
  "gemini-2.0-flash":"google/gemini-2.0-flash",
  "gemini-2.0-pro-exp":"google/gemini-2.0-pro-exp-02-05",
  "gemini-2.0-flash-thinking-exp":"google/gemini-2.0-flash-thinking-exp-1219",
  "qwq-32b":"qwen/qwq-32b",
  "qwen-max":"qwen/qwen-max"
};

// TypeScript interfaces
interface ChatMessage {
  role: string;
  content: string;
  name?: string;
}

interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  n?: number;
  stream?: boolean;
  stop?: string[] | string;
  max_tokens?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  user?: string;
}

// DeepSider API response interfaces
interface DeepSiderResponse {
  code: number;
  data?: {
    type?: string;
    content?: string;
  };
  message?: string;
}

// Helper functions
function getHeaders(apiKey: string): Record<string, string> {
  // Check if multiple tokens are provided (comma-separated)
  const tokens = apiKey.split(',');
  
  let currentToken: string;
  
  if (tokens.length > 0) {
    // Rotate tokens
    currentToken = tokens[TOKEN_INDEX % tokens.length];
    TOKEN_INDEX = (TOKEN_INDEX + 1) % tokens.length;
    logger.debug(`Using token index ${TOKEN_INDEX-1} of ${tokens.length}`);
  } else {
    currentToken = apiKey;
    logger.debug(`Using single token`);
  }
  
  return {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "content-type": "application/json",
    "origin": "chrome-extension://client",
    "i-lang": "zh-CN",
    "i-version": "1.1.64",
    "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "Windows",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "authorization": `Bearer ${currentToken.trim()}`
  };
}

function verifyApiKey(ctx: Context): string | null {
  const authHeader = ctx.request.headers.get("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    ctx.response.status = 401;
    ctx.response.body = { detail: "Invalid API key format" };
    return null;
  }
  return authHeader.replace("Bearer ", "");
}

function mapOpenaiToDeepsiderModel(model: string): string {
  return MODEL_MAPPING[model] || "anthropic/claude-3.7-sonnet";
}

function formatMessagesForDeepsider(messages: ChatMessage[]): string {
  let prompt = "";
  for (const msg of messages) {
    const role = msg.role;
    // Map OpenAI roles to DeepSider format
    if (role === "system") {
      // System messages at the beginning as guidance
      prompt = `${msg.content}\n\n` + prompt;
    } else if (role === "user") {
      prompt += `Human: ${msg.content}\n\n`;
    } else if (role === "assistant") {
      prompt += `Assistant: ${msg.content}\n\n`;
    } else {
      // Other roles treated as user
      prompt += `Human (${role}): ${msg.content}\n\n`;
    }
  }
  
  // If the last message is not from the user, add a Human prefix to prompt a response
  if (messages.length > 0 && messages[messages.length - 1].role !== "user") {
    prompt += "Human: ";
  }
  
  return prompt.trim();
}

async function generateOpenaiResponse(fullResponse: string, requestId: string, model: string): Promise<Record<string, any>> {
  const timestamp = Math.floor(Date.now() / 1000);
  return {
    "id": `chatcmpl-${requestId}`,
    "object": "chat.completion",
    "created": timestamp,
    "model": model,
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": fullResponse
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 0,  // Cannot calculate accurately
      "completion_tokens": 0,  // Cannot calculate accurately
      "total_tokens": 0  // Cannot calculate accurately
    }
  };
}

/**
 * Processes streaming responses from DeepSider API and converts them to OpenAI format
 * @param response Response object from fetch
 * @param requestId Unique request ID
 * @param model Model name
 * @param apiKey API key(s)
 * @param tokenIndex Current token index
 */
async function* streamOpenaiResponse(response: Response, requestId: string, model: string, apiKey: string, tokenIndex: number): AsyncGenerator<string> {
  const timestamp = Math.floor(Date.now() / 1000);
  let fullResponse = "";
  let messageCount = 0;
  
  try {
    logger.debug(`Starting to process streaming response for request ${requestId}`);
    
    // Get the reader from the response body
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Response body cannot be read");
    }
    
    const decoder = new TextDecoder();
    let buffer = "";
    
    // Process the stream chunk by chunk
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        logger.debug(`Stream completed for request ${requestId}`);
        break;
      }
      
      // Decode the chunk and add it to the buffer
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      
      // Process complete lines (each DeepSider message is on a separate line)
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep the last incomplete line in the buffer
      
      for (const line of lines) {
        if (!line.trim()) continue;
        
        if (line.startsWith("data: ")) {
          try {
            // Parse the JSON data from the line
            const jsonData = line.substring(6); // Remove "data: " prefix
            messageCount++;
            
            // Log raw message for debugging
            logger.debug(`Stream message #${messageCount}: ${jsonData.substring(0, 100)}${jsonData.length > 100 ? '...' : ''}`);
            
            const data: DeepSiderResponse = JSON.parse(jsonData);
            
            // Process based on DeepSider response code
            if (data.code === 202 && data.data?.type === "chat") {
              // Content message - extract the content
              const content = data.data?.content || '';
              
              if (content) {
                fullResponse += content;
                logger.debug(`Content received: "${content.substring(0, 50)}${content.length > 50 ? '...' : ''}"`);
                
                // Generate OpenAI format streaming response
                const chunk = {
                  "id": `chatcmpl-${requestId}`,
                  "object": "chat.completion.chunk",
                  "created": timestamp,
                  "model": model,
                  "choices": [
                    {
                      "index": 0,
                      "delta": {
                        "content": content
                      },
                      "finish_reason": null
                    }
                  ]
                };
                yield `data: ${JSON.stringify(chunk)}\n\n`;
              }
            } else if (data.code === 203) {
              // End of stream signal
              logger.debug(`End of stream received for request ${requestId}`);
              
              // Send completion signal
              const chunk = {
                "id": `chatcmpl-${requestId}`,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model,
                "choices": [
                  {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                  }
                ]
              };
              yield `data: ${JSON.stringify(chunk)}\n\n`;
              yield "data: [DONE]\n\n";
            } else {
              // Log other response codes
              logger.debug(`Received DeepSider response with code ${data.code}: ${JSON.stringify(data)}`);
            }
          } catch (e) {
            logger.warning(`Error parsing stream message #${messageCount}: ${e instanceof Error ? e.message : String(e)}`);
            logger.warning(`Problematic line: ${line.substring(0, 100)}${line.length > 100 ? '...' : ''}`);
          }
        } else {
          // Log unexpected format
          logger.warning(`Unexpected stream line format: ${line.substring(0, 100)}${line.length > 100 ? '...' : ''}`);
        }
      }
    }
    
    // If we never received an end signal (code 203), send one now
    if (fullResponse && messageCount > 0) {
      logger.debug(`Sending end-of-stream signal for request ${requestId} as it wasn't received from the API`);
      const chunk = {
        "id": `chatcmpl-${requestId}`,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "choices": [
          {
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
          }
        ]
      };
      yield `data: ${JSON.stringify(chunk)}\n\n`;
      yield "data: [DONE]\n\n";
    }
  } catch (e) {
    const errorMessage = e instanceof Error ? e.message : String(e);
    logger.error(`Error processing streaming response for request ${requestId}: ${errorMessage}`);
    
    // Log token information for debugging
    const tokens = apiKey.split(',');
    if (tokens.length > 1) {
      logger.info(`Using token ${tokenIndex + 1} of ${tokens.length} when error occurred`);
    }
    
    // Return error message to client
    const errorChunk = {
      "id": `chatcmpl-${requestId}`,
      "object": "chat.completion.chunk",
      "created": timestamp,
      "model": model,
      "choices": [
        {
          "index": 0,
          "delta": {
            "content": `\n\n[Error processing response: ${errorMessage}]`
          },
          "finish_reason": "stop"
        }
      ]
    };
    yield `data: ${JSON.stringify(errorChunk)}\n\n`;
    yield "data: [DONE]\n\n";
  }
}

/**
 * Processes responses from DeepSider API in non-streaming mode
 * @param response Response object from fetch
 */
async function processNonStreamingResponse(response: Response): Promise<string> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Response body cannot be read");
  }
  
  const decoder = new TextDecoder();
  let buffer = "";
  let fullResponse = "";
  let messageCount = 0;
  
  logger.debug(`Starting to process non-streaming response`);
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    
    for (const line of lines) {
      if (!line.trim()) continue;
      
      if (line.startsWith("data: ")) {
        try {
          messageCount++;
          const data: DeepSiderResponse = JSON.parse(line.substring(6));
          
          if (data.code === 202 && data.data?.type === "chat") {
            const content = data.data?.content || '';
            if (content) {
              fullResponse += content;
              logger.debug(`Content chunk #${messageCount} received, length: ${content.length}`);
            }
          }
        } catch (e) {
          logger.warning(`Error parsing non-stream message #${messageCount}: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    }
  }
  
  logger.debug(`Finished processing non-streaming response, total chunks: ${messageCount}`);
  return fullResponse;
}

// Check account balance function
async function checkAccountBalance(apiKey: string, tokenIndex: number | null = null): Promise<[boolean, Record<string, any>]> {
  const tokens = apiKey.split(',');
  
  // If token_index is provided and valid, use the specified token
  let currentToken: string;
  if (tokenIndex !== null && tokens.length > tokenIndex) {
    currentToken = tokens[tokenIndex].trim();
  } else {
    // Otherwise use the first token
    currentToken = tokens.length ? tokens[0].trim() : apiKey;
  }
  
  const headers = {
    "accept": "*/*",
    "content-type": "application/json",
    "authorization": `Bearer ${currentToken}`
  };
  
  try {
    // Get account balance info
    const response = await fetch(
      `${DEEPSIDER_API_BASE.replace('/v2', '')}/quota/retrieve`,
      { headers }
    );
    
    if (response.status === 200) {
      const data = await response.json();
      if (data.code === 0) {
        const quotaList = data.data?.list || [];
        
        // Parse balance info
        const quotaInfo: Record<string, any> = {};
        for (const item of quotaList) {
          const itemType = item.type || '';
          const available = item.available || 0;
          
          quotaInfo[itemType] = {
            "total": item.total || 0,
            "available": available,
            "title": item.title || ''
          };
        }
        
        return [true, quotaInfo];
      }
    }
    
    return [false, {}];
  } catch (e) {
    logger.warning(`Error checking account balance: ${e instanceof Error ? e.message : String(e)}`);
    return [false, {}];
  }
}

// Create router
const router = new Router();

// Routes
router.get("/", (ctx) => {
  ctx.response.body = { message: "OpenAI API Proxy service is running, connected to DeepSider API" };
});

router.get("/v1/models", async (ctx) => {
  const apiKey = verifyApiKey(ctx);
  if (!apiKey) return;
  
  const models = [];
  for (const openaiModel in MODEL_MAPPING) {
    models.push({
      "id": openaiModel,
      "object": "model",
      "created": Math.floor(Date.now() / 1000),
      "owned_by": "openai-proxy"
    });
  }
  
  ctx.response.body = {
    "object": "list",
    "data": models
  };
});

router.post("/v1/chat/completions", async (ctx) => {
  try {
    const apiKey = verifyApiKey(ctx);
    if (!apiKey) return;
    
    // Parse request body
    const body = await ctx.request.body().value;
    const chatRequest: ChatCompletionRequest = body;
    
    // Generate unique request ID
    const now = new Date();
    const timestamp = now.getTime().toString();
    const requestId = now.toISOString().replace(/[-:T.Z]/g, '').substring(0, 14) + 
                     timestamp.substring(timestamp.length - 6);
    
    logger.info(`Processing request ${requestId}, model: ${chatRequest.model}, stream: ${chatRequest.stream}`);
    
    // Map model
    const deepsiderModel = mapOpenaiToDeepsiderModel(chatRequest.model);
    logger.debug(`Mapped model ${chatRequest.model} to DeepSider model ${deepsiderModel}`);
    
    // Prepare prompt for DeepSider API
    const prompt = formatMessagesForDeepsider(chatRequest.messages);
    logger.debug(`Generated prompt (first 100 chars): ${prompt.substring(0, 100)}${prompt.length > 100 ? '...' : ''}`);
    
    // Prepare request payload
    const payload = {
      "model": deepsiderModel,
      "prompt": prompt,
      "webAccess": "close",  // Default: web access off
      "timezone": "Asia/Shanghai"
    };
    
    // Get request headers (with selected token)
    const headers = getHeaders(apiKey);
    // Get current token index
    const tokens = apiKey.split(',');
    const currentTokenIndex = tokens.length > 0 ? (TOKEN_INDEX - 1) % tokens.length : 0;
    
    logger.debug(`Sending request to DeepSider API with token index ${currentTokenIndex}`);
    
    // Send request to DeepSider API
    const response = await fetch(
      `${DEEPSIDER_API_BASE}/chat/conversation`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(payload)
      }
    );
    
    // Check response status
    if (response.status !== 200) {
      let errorMsg = `DeepSider API request failed: ${response.status}`;
      let errorDetails = "";
      
      try {
        const errorData = await response.json();
        errorDetails = errorData.message || JSON.stringify(errorData);
      } catch {
        try {
          errorDetails = await response.text();
        } catch (e) {
          errorDetails = `Could not read response: ${e instanceof Error ? e.message : String(e)}`;
        }
      }
      
      errorMsg += ` - ${errorDetails}`;
      logger.error(errorMsg);
      
      ctx.response.status = response.status;
      ctx.response.body = { 
        detail: "API request failed", 
        status: response.status,
        message: errorDetails 
      };
      return;
    }
    
    logger.debug(`Received successful response from DeepSider API for request ${requestId}`);
    
    // Handle streaming or non-streaming response
    if (chatRequest.stream) {
      logger.debug(`Processing streaming response for request ${requestId}`);
      
      // Set up streaming response
      ctx.response.type = "text/event-stream";
      
      const streamGenerator = streamOpenaiResponse(response, requestId, chatRequest.model, apiKey, currentTokenIndex);
      const body = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of streamGenerator) {
              controller.enqueue(new TextEncoder().encode(chunk));
            }
            controller.close();
            logger.debug(`Stream completed successfully for request ${requestId}`);
          } catch (e) {
            const errorMessage = e instanceof Error ? e.message : String(e);
            logger.error(`Error in stream controller for request ${requestId}: ${errorMessage}`);
            controller.error(e);
          }
        }
      });
      
      ctx.response.body = body;
    } else {
      logger.debug(`Processing non-streaming response for request ${requestId}`);
      
      // Process the full response
      const fullResponse = await processNonStreamingResponse(response);
      
      // Return OpenAI format full response
      ctx.response.body = await generateOpenaiResponse(fullResponse, requestId, chatRequest.model);
      logger.info(`Completed non-streaming request ${requestId}, response length: ${fullResponse.length}`);
    }
  } catch (e) {
    const errorMessage = e instanceof Error ? e.message : String(e);
    logger.exception(`Error processing request: ${errorMessage}`);
    
    ctx.response.status = 500;
    ctx.response.body = { 
      detail: "Internal server error", 
      error: errorMessage,
      trace: e instanceof Error ? e.stack : undefined
    };
  }
});

router.get("/admin/balance", async (ctx) => {
  // Simple admin key check
  const adminKey = ctx.request.headers.get("X-Admin-Key");
  const expectedAdminKey = env.ADMIN_KEY || "admin";
  
  if (!adminKey || adminKey !== expectedAdminKey) {
    ctx.response.status = 403;
    ctx.response.body = { detail: "Unauthorized" };
    return;
  }
  
  // Get API key from headers
  const authHeader = ctx.request.headers.get("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    ctx.response.status = 401;
    ctx.response.body = { detail: "Missing or invalid Authorization header" };
    return;
  }
  
  const apiKey = authHeader.replace("Bearer ", "");
  const tokens = apiKey.split(',');
  
  const result: Record<string, any> = {};
  
  // Get balance info for all tokens
  for (let i = 0; i < tokens.length; i++) {
    const tokenDisplay = `token_${i+1}`;
    const [success, quotaInfo] = await checkAccountBalance(apiKey, i);
    
    if (success) {
      result[tokenDisplay] = {
        "status": "success",
        "quota": quotaInfo
      };
    } else {
      result[tokenDisplay] = {
        "status": "error",
        "message": "Could not get account balance information"
      };
    }
  }
  
  ctx.response.body = result;
});

// Error handler for 404
router.all("(.*)", (ctx) => {
  ctx.response.status = 404;
  ctx.response.body = {
    "error": {
      "message": `Resource not found: ${ctx.request.url.pathname}`,
      "type": "not_found_error",
      "code": "not_found"
    }
  };
});

// Apply router
app.use(router.routes());
app.use(router.allowedMethods());

// Startup event
app.addEventListener("listen", () => {
  logger.info(`OpenAI API proxy service started, ready to accept requests`);
  logger.info(`Multiple token rotation supported, use comma-separated tokens in Authorization header`);
});

// Start server
const port = parseInt(env.PORT || "7860");
logger.info(`Starting OpenAI API proxy service on port: ${port}`);
await app.listen({ port });
