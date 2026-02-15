import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';

const OSS_TOKEN_COOKIE = 'dograh_oss_token';
const OSS_USER_COOKIE = 'dograh_oss_user';

export async function POST() {
    const cookieStore = await cookies();
    cookieStore.delete(OSS_TOKEN_COOKIE);
    cookieStore.delete(OSS_USER_COOKIE);

    return NextResponse.json({ success: true });
}
