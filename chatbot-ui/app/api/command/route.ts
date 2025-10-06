const DEFAULT_BASE_URL = "http://localhost:5000/v1"

const trimBase = (base: string) => {
  try {
    const url = new URL(base.endsWith("/") ? base : `${base}/`)
    return `${url.origin}${url.pathname.replace(/\/+$/, "")}`
  } catch {
    return DEFAULT_BASE_URL
  }
}

export const runtime = "edge"

export async function POST(request: Request) {
  const json = await request.json()
  const { input } = json as {
    input: string
  }

  try {
    const base = trimBase(process.env.OPENAI_API_BASE_URL || DEFAULT_BASE_URL)
    const response = await fetch(`${base}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.OPENAI_API_KEY
          ? { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` }
          : {})
      },
      body: JSON.stringify({
        model: "dataai",
        stream: false,
        messages: [
          { role: "system", content: "Respond to the user." },
          { role: "user", content: input }
        ],
        temperature: 0
      })
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || "Local LLM command failed")
    }

    const data = await response.json()
    const content = data?.choices?.[0]?.message?.content || ""

    return new Response(JSON.stringify({ content }), { status: 200 })
  } catch (error: any) {
    const errorMessage = error.error?.message || "An unexpected error occurred"
    const errorCode = error.status || 500
    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode
    })
  }
}
