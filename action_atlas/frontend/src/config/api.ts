// API Configuration
// Set NEXT_PUBLIC_API_URL in .env.local to point to cluster backend
// Example: NEXT_PUBLIC_API_URL=http://192.168.1.100:6006

// Use empty string for production/tunnel (relies on Next.js rewrites to proxy /api/* to backend)
// Set NEXT_PUBLIC_API_URL only for local dev when running frontend separately from backend
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

// Helper to build API URLs
export function apiUrl(path: string): string {
  const base = API_BASE_URL.replace(/\/$/, ""); // Remove trailing slash
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${base}${cleanPath}`;
}
