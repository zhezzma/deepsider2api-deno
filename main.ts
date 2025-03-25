// Import required modules
import { Application, Router, Context, Request } from "https://deno.land/x/oak@v12.6.1/mod.ts";
import { config } from "https://deno.land/x/dotenv@v3.2.2/mod.ts";

// Load environment variables
const env = config();

// Configure logging
const logger = {
  info: (message: string) => console.log(`INFO: ${new Date().toISOString()} - ${message}`),
  warning: (message: string) => console.warn(`WARNING: ${new Date().toISOString()} - ${message}`),
  error: (message: string) => console.error(`ERROR: ${new Date().toISOString()} - ${message}`),exception: (message: string) => console.error(`EXCEPTION: ${new Date().toISOString()} - ${message}`),
  debug: (message: string) => console.debug(`DEBUG: ${new Date().toISOString()} - ${message}`)
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
  }await next();
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
  stream?: boolean;stop?: string[] | string;
  max_tokens?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  user?: string;
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
    logger.debug(`Using token index ${TOKEN_INDEX-1}, from ${tokens.length} available tokens`);
  } else {
    currentToken = apiKey;
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
  const mappedModel = MODEL_MAPPING[model] || "anthropic/claude-3.7-sonnet";
  logger.debug(`Model mapping: ${model} => ${mappedModel}`);
  return mappedModel;
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
  if (messages.length > 0&& messages[messages.length - 1].role !== "user") {
    prompt += "Human: ";
  }
  
  logger.debug(`Formatted prompt: ${prompt.length} chars`);
  return prompt.trim();
}

async function generateOpenaiResponse(fullResponse: string, requestId: string, model: string): Promise<Record<string, any>> {
  const timestamp = Math.floor(Date.now() / 1000);
  logger.debug(`Generating non-streaming response with id: ${requestId}`);
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
 * 处理流式响应数据
 *从DeepSider API获取流式响应并转换为OpenAI API格式的流
 *
 * @param response DeepSider API的响应对象
 * @param requestId 唯一的请求ID
 * @param model 模型名称
 * @param apiKey API密钥(可能包含多个token)
 * @param tokenIndex 当前使用的token索引
 */
async function* streamOpenaiResponse(response: Response, requestId: string, model: string, apiKey: string, tokenIndex: number): AsyncGenerator<string> {
  const timestamp = Math.floor(Date.now() / 1000);
  let fullResponse = "";
  let chunkCounter = 0;
  let totalBytes = 0;
  
  logger.info(`开始处理流式响应: requestId=${requestId}, model=${model}, tokenIndex=${tokenIndex}`);
  
  try {
    // 1. 获取响应体的Reader
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("无法读取响应体");
    }// 2. 设置解码器和缓冲区
    const decoder = new TextDecoder();
    let buffer = "";
    
    // 3. 读取并处理数据流
    while (true) {
      // 3.1 读取数据块
      const { done, value } = await reader.read();
      if (done) {
        logger.debug(`流读取完成: requestId=${requestId}`);
        break;
      }
      // 3.2 解码数据并添加到缓冲区
      totalBytes += value.length;
      buffer += decoder.decode(value, { stream: true });
      
      // 3.3 处理完整行
      const lines = buffer.split("\n");
      // 保留最后一个可能不完整的行在缓冲区
      buffer = lines.pop() || "";
      
      for (const line of lines) {
        if (!line.trim()) continue;
        if (line.startsWith("data: ")) {
          try {
            // 3.4 解析JSON数据
            const data = JSON.parse(line.substring(6));
            logger.debug(`接收到数据: code=${data.code}, type=${data.data?.type ||'unknown'}`);
            
            if (data.code === 202&& data.data?.type === "chat") {
              // 接收到聊天内容
              const content = data.data?.content || '';
              if (content) {
                fullResponse += content;chunkCounter++;
                // 3.5 生成OpenAI格式的流式响应
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
                
                // 每10个chunk记录一次日志
                if (chunkCounter % 10 === 0) {
                  logger.debug(`已处理 ${chunkCounter} 个数据块, 总计 ${fullResponse.length} 字符`);
                }
                
                yield `data: ${JSON.stringify(chunk)}\n\n`;
              }
            } else if (data.code === 203) {
              // 3.6 接收到完成信号
              logger.info(`接收到完成信号: requestId=${requestId},总计 ${chunkCounter} 个数据块, ${fullResponse.length} 字符`);
              
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
              yielJSON.stringify(chunk)}\n\n`;yield "data: [DONE]\n\n";
            }
          } catch (e) {
            // 3.7 处理JSON解析错误
            logger.warning(`无法解析响应行: ${line}\n错误: ${e instanceof Error ? e.message : String(e)}`);
          }
        }
      }
    }
    
    // 4. 确保最后的缓冲区也被处理
    if (buffer.trim()) {
      logger.debug(`处理剩余缓冲区: ${buffer.length} 字符`);
      if (buffer.start ")) {
        try {
          const data = JSON.parse(buffer.substring(6));
          if (data.code === 202 && data.data?.type === "chat") {
            const content = data.data?.content || '';
            if (content) {
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
              yielJSON.stringify(chunk)}\n\n`;
            }
          }
        } catch (e) {
          logger.warning(`处理剩余缓冲区时出错: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    }
    
    // 5. 发送最终的结束信号(以防未收到203信号)
    if (buffer.indexOf('"code":203') === -1 && buffer.indexOf("DONE]") === -1) {
      logger.debug(`未检测到正常结束信号，发送兜底结束信号`);
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
       [DONE]\n\n";
    }} catch (e) {
    // 6. 处理流处理过程中的错误
    const errorMsg = e instanceof Error ? e.message : String(e);
    logger.error(`处理流式响应时出错: ${errorMsg}`);
    logger.error(`错误详情: requestId=${requestId}, 已接收字节=${totalBytes}, 已处理块=${chunkCounter}`);
    
    // 7. 尝试使用下一个token(仅记录日志，不实现自动重试)
    const tokens = apiKey.split(',');
    if (tokens.length > 1) {
      logger.info(`有多个token可用，可以考虑使用下一个token重试`);}
    
    // 8. 返回错误消息
    const errorChunk = {
      "id": `chatcmpl-${requestId}`,
      "object": "chat.completion.chunk",
      "created": timestamp,
      "model": model,
      "choices": [
        {
          "index": 0,
          "delta": {
            "content": `\n\n[处理响应时发生错误: ${errorMsg}]`
          },
          "finish_reason": "stop"
        }
      ]
    };
    yield `data: ${JSON.stringify(errorChunk)}\n\n`; [DONE]\n\n";
  }
}

/**
 * 处理非流式响应数据
 * 从DeepSider API获取内容并转换为OpenAI API格式
 *
 * @param response DeepSider API的响应对象
 * @param requestId 唯一的请求ID
 * @returns 完整的响应内容
 */
async function processFullResponse(response: Response, requestId: string): Promise<string> {
  logger.debug(`处理完整(非流式)响应: requestId=${requestId}`);
  
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("无法读取响应体");
  }
  
  const decoder = new TextDecoder();
  let buffer = "";
  let fullResponse = "";
  let chunkCounter = 0;
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    
    for (const line of lines) {
      if (!line.trim()) continue;if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.substring(6));
          if (data.code === 202 && data.data?.type === "chat") {
            const content = data.data?.content || '';
            if (content) {
              fullResponse += content;
              chunkCounter++;
              
              // 每10个chunk记录一次日志
              if (chunkCounter % 10 === 0) {
                logger.debug(`非流式响应:已接收 ${chunkCounter} 个数据块, 总计 ${fullResponse.length} 字符`);
              }}
          }
        } catch (e) {
          logger.warning(`解析非流式响应行出错: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    }
  }
  
  // 处理最后的缓冲区
  if (buffer.trim() && buffer.startsWith")) {
    try {
      const data = JSON.parse(buffer.substring(6));
      if (data.code === 202 && data.data?.type === "chat") {
        const content = data.data?.content || '';
        if (content) {
          fullResponse += content;
        }
      }
    } catch (e) {
      logger.warning(`处理最后缓冲区时出错: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  
  logger.info(`完整响应接收完毕: requestId=${requestId}, 数据块=${chunkCounter}, 内容长度=${fullResponse.length}`);
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
  const startTime = Date.now();
  const apiKey = verifyApiKey(ctx);
  if (!apiKey) return;
  
  // 记录请求开始
  const requestStartMsg = `开始处理API请求: ${ctx.request.url.pathname}`;
  logger.info(requestStartMsg);
  
  try {
    // Parse request body
    const body = await ctx.request.body().value;
    const chatRequest: ChatCompletionRequest = body;
    
    // 记录请求基本信息
    logger.debug(`请求内容: model=${chatRequest.model}, messages=${chatRequest.messages.length}条, stream=${chatRequest.stream}`);
    
    // Generate unique request ID
    const now = new Date();
    const timestamp = now.getTime().toString();
    const requestId = now.toISOString().replace(/[-:T.Z]/g, '').substring(0, 14) + timestamp.substring(timestamp.length - 6);
    
    // Map model
    const deepsiderModel = mapOpenaiToDeepsiderModel(chatRequest.model);
    
    // Prepare prompt for DeepSider API
    const prompt = formatMessagesForDeepsider(chatRequest.messages);// Prepare request payload
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
    const currentTokenIndex = tokens.length > 0 ? (TOKEN_INDEX - 1 + tokens.length) % tokens.length : 0;
    
    logger.info(`发送请求到DeepSider API: requestId=${requestId}, model=${deepsiderModel}, tokenIndex=${currentTokenIndex}`);
    
    // Send request to DeepSider API
    const response = await fetch(
      `${DEEPSIDER_API_BASE}/chat/conversation`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(payload)
      }
    );
    
    // 记录API响应状态
    logger.debug(`DeepSider API响应: status=${response.status}, requestId=${requestId}`);
    
    // Check response status
    if (response.status !== 200) {
      let errorMsg = `DeepSider API请求失败: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMsg +=` - ${errorData.message || '未知错误'}`;
      } catch {
        errorMsg += ` - ${await response.text()}`;
      }
      logger.error(errorMsg);
      ctx.response.status = response.status;
      ctx.response.body = { detail: "API request failed", message: errorMsg };return;
    }
    
    // 根据请求类型处理响应
    if (chatRequest.stream) {
      logger.info(`处理流式响应: requestId=${requestId}`);
      
      // Set up streaming response
      ctx.response.type = "text/event-stream";
      
      const streamGenerator = streamOpenaiResponse(response, requestId, chatRequest.model, apiKey, currentTokenIndex);
      const body = new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of streamGenerator) {
              controller.enqueue(new TextEncoder().encode(chunk));
            }controller.close();
          } catch (e) {
            const errorMsg = e instanceof Error ? e.message : String(e);
            logger.error(`流式响应处理错误: ${errorMsg}`);
            controller.error(e);
          }
        }
      });
      
      ctx.response.body = body;
    } else {
      logger.info(`处理非流式响应: requestId=${requestId}`);// 获取完整响应
      const fullResponse = await processFullResponse(response, requestId);
      
      // 返回OpenAI格式的完整响应
      ctx.response.body = await generateOpenaiResponse(fullResponse, requestId, chatRequest.model);
    }
    // 记录请求完成时间
    const endTime = Date.now();
    logger.info(`请求处理完成: requestId=${requestId}, 耗时=${endTime - startTime}ms`);} catch (e) {
    const errorMsg = e instanceof Error ? e.message : String(e);
    logger.exception(`处理请求时发生异常: ${errorMsg}`);
    logger.error(e instanceof Error && e.stack ? e.stack : '无堆栈信息');
    
    ctx.response.status = 500;
    ctx.response.body = { 
      detail: `Internal server error`, 
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
    // 记录请求失败时间
    const endTime = Date.now();
    logger.info(`请求处理失败: 耗时=${endTime - startTime}ms`);
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
    const [success, quotaInfo] = await checkAccountBalance(apiKey, i);if (success) {
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

// Error handler for404
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
  logger.info(`OpenAI API proxy service started, ready to accept requests`);logger.info(`Multiple token rotation supported, use comma-separated tokens in Authorization header`);
});

// Start server
const port = parseInt(env.PORT || "7860");
logger.info(`Starting OpenAI API proxy service on port: ${port}`);
await app.listen({ port });
