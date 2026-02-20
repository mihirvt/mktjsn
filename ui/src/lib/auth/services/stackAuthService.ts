'use client';

import type { CurrentUser } from '@stackframe/stack';

import logger from '@/lib/logger';

import type { IAuthService } from './interface';

export class StackAuthService implements IAuthService {
  private userInstance: CurrentUser | null = null;
  private callCount = {
    setUserInstance: 0,
    getAccessToken: 0,
    refreshToken: 0,
    getCurrentUser: 0,
    isAuthenticated: 0
  };

  // Set the user instance from the Stack useUser hook
  setUserInstance(user: CurrentUser) {
    this.callCount.setUserInstance++;
    logger.debug('[StackAuthService] setUserInstance called', {
      callCount: this.callCount.setUserInstance,
      userId: user?.id,
      hadPreviousUser: !!this.userInstance,
      previousUserId: this.userInstance?.id,
      timestamp: new Date().toISOString()
    });
    this.userInstance = user;
    logger.debug('[StackAuthService] setUserInstance completed - user is now set');
  }

  async getAccessToken(): Promise<string> {
    this.callCount.getAccessToken++;
    logger.debug('[StackAuthService] getAccessToken called', {
      callCount: this.callCount.getAccessToken,
      hasUser: !!this.userInstance,
      userId: this.userInstance?.id
    });

    if (!this.userInstance) {
      logger.error('[StackAuthService] getAccessToken - User not initialized');
      throw new Error('User not initialized');
    }

    logger.debug('[StackAuthService] Calling user.getAuthJson()');
    const authJson = await this.userInstance.getAuthJson();
    logger.debug('[StackAuthService] getAuthJson returned', {
      hasToken: !!authJson.accessToken,
      tokenLength: authJson.accessToken?.length
    });

    if (!authJson.accessToken) {
      logger.error('[StackAuthService] No access token available');
      throw new Error('No access token available');
    }
    return authJson.accessToken;
  }

  async refreshToken(): Promise<string> {
    this.callCount.refreshToken++;
    logger.debug('[StackAuthService] refreshToken called', {
      callCount: this.callCount.refreshToken,
      hasUser: !!this.userInstance
    });

    if (!this.userInstance) {
      throw new Error('User not initialized');
    }
    // Stack handles token refresh internally
    const authJson = await this.userInstance.getAuthJson();
    if (!authJson.accessToken) {
      throw new Error('No access token available');
    }
    return authJson.accessToken;
  }

  async getCurrentUser(): Promise<CurrentUser | null> {
    this.callCount.getCurrentUser++;
    logger.debug('[StackAuthService] getCurrentUser called', {
      callCount: this.callCount.getCurrentUser,
      hasUser: !!this.userInstance,
      userId: this.userInstance?.id
    });
    // Return the actual Stack user instance
    return this.userInstance;
  }

  isAuthenticated(): boolean {
    this.callCount.isAuthenticated++;
    const isAuth = !!this.userInstance;
    logger.debug('[StackAuthService] isAuthenticated called', {
      callCount: this.callCount.isAuthenticated,
      result: isAuth,
      hasUserInstance: !!this.userInstance,
      userId: this.userInstance?.id,
      timestamp: new Date().toISOString()
    });
    return isAuth;
  }

  redirectToLogin(): void {
    if (typeof window !== 'undefined') {
      window.location.href = '/handler/sign-in';
    }
  }

  async logout(): Promise<void> {
    if (this.userInstance && this.userInstance.signOut) {
      await this.userInstance.signOut();
    }
  }

  getSelectedTeam(): unknown {
    return this.userInstance?.selectedTeam;
  }

  async listPermissions(team?: unknown): Promise<Array<{ id: string }>> {
    if (!this.userInstance || !this.userInstance.listPermissions) {
      return [];
    }
    const targetTeam = team || this.userInstance.selectedTeam;
    if (!targetTeam) {
      return [];
    }
    try {
      const perms = await this.userInstance.listPermissions(targetTeam);
      return Array.isArray(perms) ? perms : [];
    } catch (error) {
      logger.error('Error listing permissions:', error);
      return [];
    }
  }

  getProviderName(): string {
    return 'stack';
  }
}

