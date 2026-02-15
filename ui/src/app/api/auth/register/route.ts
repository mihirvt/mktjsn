import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';

const OSS_TOKEN_COOKIE = 'dograh_oss_token';
const OSS_USER_COOKIE = 'dograh_oss_user';

export async function POST(request: Request) {
    const { email, password } = await request.json();
    const backendUrl = process.env.BACKEND_URL || 'http://api:8000';

    try {
        const response = await fetch(`${backendUrl}/api/v1/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        if (response.ok) {
            const data = await response.json();
            const token = data.access_token;

            // Fetch user profile from backend to store in cookie
            const userResponse = await fetch(`${backendUrl}/api/v1/user/config`, {
                headers: { Authorization: `Bearer ${token}` },
            });

            if (userResponse.ok) {
                const userData = await userResponse.json();
                const user = {
                    id: userData.id,
                    name: userData.email,
                    provider: 'local',
                    organizationId: userData.selected_organization_id,
                };

                const cookieStore = await cookies();
                cookieStore.set(OSS_TOKEN_COOKIE, token, {
                    httpOnly: true,
                    secure: process.env.NODE_ENV === 'production',
                    sameSite: 'lax',
                    maxAge: 60 * 60 * 24 * 30, // 30 days
                    path: '/',
                });

                cookieStore.set(OSS_USER_COOKIE, JSON.stringify(user), {
                    httpOnly: true,
                    secure: process.env.NODE_ENV === 'production',
                    sameSite: 'lax',
                    maxAge: 60 * 60 * 24 * 30, // 30 days
                    path: '/',
                });

                return NextResponse.json({ success: true, user });
            }
        }

        const errorData = await response.json();
        return NextResponse.json({ error: errorData.detail || 'Registration failed' }, { status: response.status });
    } catch (error) {
        console.error('Registration error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
    }
}
