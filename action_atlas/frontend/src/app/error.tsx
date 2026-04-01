'use client'

import { useEffect } from 'react'

interface ErrorProps {
  error: Error & { digest?: string }
  reset: () => void
}

export default function Error({ error, reset }: ErrorProps) {
  useEffect(() => {
    // Log the error for debugging
    console.error('App error caught by error boundary:', error)
  }, [error])

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        backgroundColor: '#f5f5f5',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}
    >
      <div
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          padding: '40px',
          maxWidth: '500px',
          textAlign: 'center',
        }}
      >
        <h1
          style={{
            fontSize: '28px',
            fontWeight: '600',
            color: '#d32f2f',
            margin: '0 0 16px 0',
          }}
        >
          Something went wrong
        </h1>

        <p
          style={{
            fontSize: '16px',
            color: '#666',
            margin: '0 0 24px 0',
            lineHeight: '1.5',
          }}
        >
          An unexpected error occurred. Please try again.
        </p>

        {error.message && (
          <details
            style={{
              marginBottom: '24px',
              padding: '12px',
              backgroundColor: '#f9f9f9',
              borderRadius: '4px',
              border: '1px solid #e0e0e0',
              textAlign: 'left',
              cursor: 'pointer',
            }}
          >
            <summary style={{ fontWeight: '500', color: '#424242' }}>
              Error details
            </summary>
            <pre
              style={{
                margin: '12px 0 0 0',
                fontSize: '12px',
                color: '#666',
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word',
              }}
            >
              {error.message}
            </pre>
          </details>
        )}

        <div
          style={{
            display: 'flex',
            gap: '12px',
            justifyContent: 'center',
          }}
        >
          <button
            onClick={reset}
            style={{
              padding: '10px 20px',
              fontSize: '14px',
              fontWeight: '500',
              backgroundColor: '#1976d2',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'background-color 0.2s',
            }}
            onMouseOver={(e) =>
              (e.currentTarget.style.backgroundColor = '#1565c0')
            }
            onMouseOut={(e) =>
              (e.currentTarget.style.backgroundColor = '#1976d2')
            }
          >
            Try Again
          </button>

          <button
            onClick={() => window.location.reload()}
            style={{
              padding: '10px 20px',
              fontSize: '14px',
              fontWeight: '500',
              backgroundColor: '#757575',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'background-color 0.2s',
            }}
            onMouseOver={(e) =>
              (e.currentTarget.style.backgroundColor = '#616161')
            }
            onMouseOut={(e) =>
              (e.currentTarget.style.backgroundColor = '#757575')
            }
          >
            Reload Page
          </button>
        </div>
      </div>
    </div>
  )
}
