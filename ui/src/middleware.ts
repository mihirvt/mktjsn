import type { NextRequest } from 'next/server';
import { NextResponse } from 'next/server';

const OSS_TOKEN_COOKIE = 'dograh_oss_token';
const OSS_USER_COOKIE = 'dograh_oss_user';

function generateOSSToken(): string {
  return `oss_${Date.now()}_${crypto.randomUUID()}`;
}

export function middleware(request: NextRequest) {
  const authProvider = process.env.NEXT_PUBLIC_AUTH_PROVIDER || 'stack';

  // Only handle OSS mode
  if (authProvider !== 'local') {
    return NextResponse.next();
  }

  // If no token exists, but we are in local mode, check if we should redirect to sign-in
  const isSignInPage = request.nextUrl.pathname === '/sign-in' || request.nextUrl.pathname === '/sign-up';

  if (!token && !isSignInPage) {
    // For now, let's check an env var or just default to showing sign-in if we want login enabled
    if (process.env.NEXT_PUBLIC_LOCAL_AUTH_ENABLED === 'true') {
      const url = request.nextUrl.clone();
      url.pathname = '/sign-in';
      return NextResponse.redirect(url);
    }

    // Fallback to legacy automatic token generation
    const response = NextResponse.next();
    const newToken = generateOSSToken();
    const user = {
      id: newToken,
      name: 'Local User',
      provider: 'local',
      organizationId: `org_${newToken}`,
    };

    response.cookies.set(OSS_TOKEN_COOKIE, newToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 30, // 30 days
      path: '/',
    });

    response.cookies.set(OSS_USER_COOKIE, JSON.stringify(user), {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 30, // 30 days
      path: '/',
    });
    return response;
  }

  return NextResponse.next();
}

// Configure which routes the middleware runs on
export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - api routes
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public files (public folder)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|public).*)',
  ],
};
