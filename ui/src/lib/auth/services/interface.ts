import type { AuthUser } from '../types';

export interface IAuthService {
  // Token management
  getAccessToken(): Promise<string>;
  refreshToken(): Promise<string>;

  // User management
  getCurrentUser(): Promise<AuthUser | null>;
  isAuthenticated(): boolean;

  // Navigation
  redirectToLogin(): void;
  logout(): Promise<void>;

  // Team/Organization management (optional for some providers)
  getSelectedTeam?(): unknown;
  listPermissions?(team?: unknown): Promise<Array<{ id: string }>>;

  // Provider-specific
  getProviderName(): string;
}

