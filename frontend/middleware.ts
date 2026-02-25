import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const CAPACITY_LOCK_ENABLED =
  process.env.NEXT_PUBLIC_CAPACITY_LOCK === "true" ||
  process.env.CAPACITY_LOCK === "true";

export function middleware(request: NextRequest) {
  if (!CAPACITY_LOCK_ENABLED) {
    return NextResponse.next();
  }

  const url = request.nextUrl.clone();
  url.pathname = "/examples";
  url.search = "";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/signin/:path*", "/upload/:path*", "/dashboard/:path*"],
};
