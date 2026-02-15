'use client';

import { LocalSignUpForm } from '@/components/auth/LocalSignUpForm';
import Footer from '@/components/Footer';

export default function SignUpPage() {
    return (
        <div className="flex min-h-screen items-center justify-center p-4">
            <LocalSignUpForm />
            <Footer />
        </div>
    );
}
