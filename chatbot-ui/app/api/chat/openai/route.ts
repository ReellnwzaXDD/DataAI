import { ChatSettings } from "@/types"
import { ServerRuntime } from "next"
import { StreamingTextResponse } from "ai"

const DEFAULT_BASE_URL = "http://localhost:5000/v1"

const trimTrailingSlashes = (path: string) => {
  let result = path
  while (result.length > 1 && result.endsWith("/")) {
    result = result.slice(0, -1)
  }
  return result
}

const buildUrl = (base: string) => {
  try {
    const url = new URL(base.endsWith("/") ? base : `${base}/`)
    const trimmedPath = trimTrailingSlashes(url.pathname)
    url.pathname = `${trimmedPath}/chat/completions`
    return url.toString()
  } catch {
    const fallback = new URL("/chat/completions", DEFAULT_BASE_URL)
    return fallback.toString()
  }
}

interface StreamState {
  inside: boolean
}

const stripThinkContent = (input: string, state: StreamState) => {
  if (!input) {
    return ""
  }

  const openTag = "<think>"
  const closeTag = "</think>"
  let output = ""
  let cursor = 0

  while (cursor < input.length) {
    if (!state.inside) {
      const nextOpen = input.indexOf(openTag, cursor)
      const nextClose = input.indexOf(closeTag, cursor)

      if (nextClose !== -1 && (nextOpen === -1 || nextClose < nextOpen)) {
        // stray closing tag; skip it
        cursor = nextClose + closeTag.length
        state.inside = false
        continue
      }

      if (nextOpen === -1) {
        output += input.slice(cursor)
        break
      }

      output += input.slice(cursor, nextOpen)
      cursor = nextOpen + openTag.length
      state.inside = true
    } else {
      const nextClose = input.indexOf(closeTag, cursor)
      if (nextClose === -1) {
        // remain inside think until next chunk
        cursor = input.length
        break
      }
      cursor = nextClose + closeTag.length
      state.inside = false
    }
  }

  return output
}

export const runtime: ServerRuntime = "edge"

export async function POST(request: Request) {
  const json = await request.json()
  const { chatSettings, messages } = json as {
    chatSettings: ChatSettings
    messages: any[]
  }

  const baseUrl = process.env.OPENAI_API_BASE_URL || DEFAULT_BASE_URL
  const target = buildUrl(baseUrl)

  const headers: Record<string, string> = {
    "Content-Type": "application/json"
  }

  const apiKey = process.env.OPENAI_API_KEY
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`
  }

  try {
    const upstream = await fetch(target, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: chatSettings.model,
        messages,
        temperature: chatSettings.temperature,
        stream: true
      })
    })

    if (!upstream.body) {
      const text = await upstream.text()
      return new Response(
        text || JSON.stringify({ message: "Local LLM returned no body" }),
        { status: upstream.status || 500 }
      )
    }

    if (!upstream.ok) {
      const text = await upstream.text()
      return new Response(
        text || JSON.stringify({ message: "Local LLM request failed" }),
        { status: upstream.status }
      )
    }

    const decoder = new TextDecoder()
    const encoder = new TextEncoder()
    const state: StreamState = { inside: false }

    const textStream = new ReadableStream<Uint8Array>({
      async start(controller) {
        const reader = upstream.body!.getReader()
        let buffer = ""
        let accumulated = ""

        const enqueueDiff = (nextContent: string) => {
          if (!nextContent) return
          if (nextContent.length <= accumulated.length) return
          const diff = nextContent.slice(accumulated.length)
          accumulated = nextContent
          controller.enqueue(encoder.encode(diff))
        }

        const processEvent = (event: string) => {
          if (!event || event.startsWith(":")) {
            return false
          }
          if (!event.startsWith("data:")) {
            return false
          }
          const payload = event.slice(5).trim()
          if (!payload) return false
          if (payload === "[DONE]") {
            return true
          }
          try {
            const parsed = JSON.parse(payload)
            if (Array.isArray(parsed?.choices)) {
              for (const choice of parsed.choices) {
                const delta = choice?.delta
                if (delta && typeof delta === "object" && typeof delta.content === "string") {
                  const cleaned = stripThinkContent(delta.content, state)
                  if (cleaned) {
                    enqueueDiff(accumulated + cleaned)
                  }
                }
                const finishReason = choice?.finish_reason
                if (finishReason && accumulated.length > 0 && !choice?.delta?.content) {
                  enqueueDiff(accumulated)
                }
              }
            }
          } catch (err) {
            console.warn("Failed to parse chunk", err)
          }
          return false
        }

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              buffer += decoder.decode()
              while (buffer.length) {
                const idx = buffer.indexOf("\n\n")
                if (idx === -1) break
                const event = buffer.slice(0, idx)
                buffer = buffer.slice(idx + 2)
                const finished = processEvent(event)
                if (finished) {
                  controller.close()
                  reader.releaseLock()
                  return
                }
              }
              break
            }

            buffer += decoder.decode(value, { stream: true })
            let idx
            while ((idx = buffer.indexOf("\n\n")) !== -1) {
              const event = buffer.slice(0, idx)
              buffer = buffer.slice(idx + 2)
              const finished = processEvent(event)
              if (finished) {
                controller.close()
                reader.releaseLock()
                return
              }
            }
          }
        } catch (error) {
          controller.error(error)
          reader.releaseLock()
          return
        }

        controller.close()
        reader.releaseLock()
      }
    })

    return new StreamingTextResponse(textStream)
  } catch (error: any) {
    return new Response(
      JSON.stringify({
        message: error?.message || "Failed to contact local LLM"
      }),
      {
        status: 500
      }
    )
  }
}
