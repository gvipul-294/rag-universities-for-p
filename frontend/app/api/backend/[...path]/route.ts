import { NextRequest, NextResponse } from "next/server";

const allowedPaths = new Set([
  "api/v1/ask",
  "api/v1/match",
  "api/v1/health",
  "api/v1/index/status",
]);

type RouteContext = { params: Promise<{ path: string[] }> };

async function proxy(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  const endpoint = path.join("/");
  if (!allowedPaths.has(endpoint)) {
    return NextResponse.json({ detail: "Backend endpoint is not allowed." }, { status: 404 });
  }

  const baseUrl = process.env.PHD_RAG_API_URL?.replace(/\/+$/, "");
  if (!baseUrl) {
    return NextResponse.json(
      { detail: "PHD_RAG_API_URL is not configured on the frontend server." },
      { status: 503 },
    );
  }

  const headers = new Headers({ Accept: "application/json" });
  if (request.method === "POST") headers.set("Content-Type", "application/json");
  const token = process.env.PHD_RAG_API_TOKEN;
  if (token && endpoint !== "api/v1/health") {
    headers.set("Authorization", `Bearer ${token}`);
  }

  try {
    const response = await fetch(`${baseUrl}/${endpoint}`, {
      method: request.method,
      headers,
      body: request.method === "POST" ? await request.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(60_000),
    });
    return new NextResponse(response.body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("Content-Type") ?? "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return NextResponse.json(
      { detail: "Could not reach the PhD Finder API. Check its URL and availability." },
      { status: 502 },
    );
  }
}

export async function GET(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}
