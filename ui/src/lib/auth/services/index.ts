import logger from '@/lib/logger';

import type { AuthProvider } from '../types';
import type { IAuthService } from './interface';
import { LocalAuthService } from './localAuthService';
import { StackAuthService } from './stackAuthService';

// Singleton instances for auth services
let stackServiceInstance: StackAuthService | null = null;
let localServiceInstance: LocalAuthService | null = null;

export function createAuthService(provider?: AuthProvider | string): IAuthService {
  const authProvider = provider || process.env.NEXT_PUBLIC_AUTH_PROVIDER || 'stack';

  switch (authProvider) {
    case 'stack':
      if (!stackServiceInstance) {
        logger.debug('[createAuthService] Creating singleton StackAuthService instance');
        stackServiceInstance = new StackAuthService();
      }
      return stackServiceInstance;
    case 'local':
      if (!localServiceInstance) {
        logger.debug('[createAuthService] Creating singleton LocalAuthService instance');
        localServiceInstance = new LocalAuthService();
      }
      return localServiceInstance;
    // Future providers can be added here
    // case 'auth0':
    //   return new Auth0Service();
    // case 'supabase':
    //   return new SupabaseService();
    default:
      console.warn(`Unknown auth provider: ${authProvider}, falling back to local`);
      if (!localServiceInstance) {
        localServiceInstance = new LocalAuthService();
      }
      return localServiceInstance;
  }
}

export type { IAuthService } from './interface';
export { LocalAuthService } from './localAuthService';
export { StackAuthService } from './stackAuthService';

