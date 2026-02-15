'use client';

import { Loader2 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { createAuthService } from '@/lib/auth/services';
import { LocalAuthService } from '@/lib/auth/services/localAuthService';

export function LocalSignInForm() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const router = useRouter();
    const authService = createAuthService('local') as LocalAuthService;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        try {
            const success = await authService.login(email, password);
            if (success) {
                toast.success('Logged in successfully');
                router.push('/dashboard'); // or wherever the home page is
                router.refresh();
            } else {
                toast.error('Invalid email or password');
            }
        } catch {
            toast.error('An error occurred during login');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="w-full max-w-md mx-auto">
            <CardHeader>
                <CardTitle>Sign In</CardTitle>
                <CardDescription>Enter your email and password to access your account.</CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmit}>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="email">Email</Label>
                        <Input
                            id="email"
                            type="email"
                            placeholder="name@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="password">Password</Label>
                        <Input
                            id="password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                </CardContent>
                <CardFooter className="flex flex-col space-y-2">
                    <Button type="submit" className="w-full" disabled={loading}>
                        {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : 'Sign In'}
                    </Button>
                    <div className="text-sm text-center text-gray-500">
                        Don&apos;t have an account?{' '}
                        <button
                            type="button"
                            onClick={() => router.push('/sign-up')}
                            className="text-primary hover:underline"
                        >
                            Sign Up
                        </button>
                    </div>
                </CardFooter>
            </form>
        </Card>
    );
}
