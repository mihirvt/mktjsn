'use client';

import logger from '@/lib/logger';

import type { LocalUser } from '../types';
import type { IAuthService } from './interface';

export class LocalAuthService implements IAuthService {
  private currentUser: LocalUser | null = null;
  private currentToken: string | null = null;
  private authPromise: Promise<void> | null = null;
  private static instance: LocalAuthService | null = null;

  constructor() {
    // Singleton pattern to ensure single initialization
    if (LocalAuthService.instance) {
      return LocalAuthService.instance;
    }
    LocalAuthService.instance = this;

    // Initialize auth on creation
    if (typeof window !== 'undefined') {
      this.authPromise = this.initializeAuth();
    }
  }

  private async initializeAuth(): Promise<void> {
    if (typeof window === 'undefined') return;

    try {
      // Fetch current auth state from our API (which reads cookies)
      const response = await fetch('/api/auth/oss');
      if (response.ok) {
        const data = await response.json();
        this.currentToken = data.token;
        this.currentUser = data.user;
        logger.info('Local auth initialized', { user: data.user });
      } else {
        logger.error('Failed to initialize local auth');
      }
    } catch (error) {
      logger.error('Error initializing local auth', error);
    }
  }

  async login(email: string, password: string): Promise<boolean> {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        this.currentToken = data.token;
        this.currentUser = data.user;
        return true;
      }
      return false;
    } catch (error) {
      logger.error('Login error', error);
      return false;
    }
  }

  async register(email: string, password: string): Promise<boolean> {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        this.currentToken = data.token;
        this.currentUser = data.user;
        return true;
      }
      return false;
    } catch (error) {
      logger.error('Registration error', error);
      return false;
    }
  }

  private async ensureAuth(): Promise<void> {
    if (this.authPromise) {
      await this.authPromise;
    } else if (!this.currentToken && typeof window !== 'undefined') {
      this.authPromise = this.initializeAuth();
      await this.authPromise;
    }
  }

  async getAccessToken(): Promise<string> {
    if (typeof window === 'undefined') {
      // SSR: Server will handle this
      return 'ssr-placeholder-token';
    }

    await this.ensureAuth();

    // If token is still null, try re-initializing (e.g., after login set cookies)
    if (!this.currentToken) {
      this.authPromise = this.initializeAuth();
      await this.authPromise;
    }

    if (!this.currentToken) {
      logger.warn('No OSS token available after initialization');
      return '';
    }
    return this.currentToken;
  }

  async refreshToken(): Promise<string> {
    // For local mode, just return the same token
    return this.getAccessToken();
  }

  async getCurrentUser(): Promise<LocalUser | null> {
    if (typeof window === 'undefined') {
      // SSR: Server will handle this
      return null;
    }

    await this.ensureAuth();

    if (!this.currentUser) {
      logger.warn('No OSS user available after initialization');
      return null;
    }

    return this.currentUser;
  }

  isAuthenticated(): boolean {
    return !!this.currentToken;
  }

  redirectToLogin(): void {
    // No-op for local mode
    logger.info('Login redirect not needed in local mode');
  }

  async logout(): Promise<void> {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
    } catch (error) {
      logger.error('Logout error', error);
    }
    this.currentUser = null;
    this.currentToken = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('dograh_auth_token');
      localStorage.removeItem('dograh_auth_user');
    }
    logger.info('Logged out from local mode');
  }

  getProviderName(): string {
    return 'local';
  }
}

