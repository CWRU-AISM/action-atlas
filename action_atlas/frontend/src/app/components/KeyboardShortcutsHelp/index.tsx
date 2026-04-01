"use client";
import React, { useEffect, useCallback } from "react";
import { Box, Typography, IconButton, Paper, Fade, Backdrop } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import KeyboardIcon from "@mui/icons-material/Keyboard";

interface KeyboardShortcutsHelpProps {
  open: boolean;
  onClose: () => void;
}

interface ShortcutItem {
  keys: string[];
  description: string;
}

interface ShortcutCategory {
  title: string;
  icon?: React.ReactNode;
  shortcuts: ShortcutItem[];
}

// Keyboard key component with nice styling
const KeyboardKey: React.FC<{ children: React.ReactNode; wide?: boolean }> = ({
  children,
  wide = false,
}) => (
  <Box
    component="kbd"
    sx={{
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      minWidth: wide ? 60 : 28,
      height: 28,
      px: 1,
      backgroundColor: "#1e293b",
      border: "1px solid #334155",
      borderRadius: "6px",
      boxShadow: "0 2px 0 #0f172a, 0 3px 3px rgba(0,0,0,0.3)",
      color: "#e2e8f0",
      fontSize: "11px",
      fontFamily: "system-ui, -apple-system, sans-serif",
      fontWeight: 600,
      textTransform: "capitalize",
      marginRight: "4px",
      "&:last-child": {
        marginRight: 0,
      },
    }}
  >
    {children}
  </Box>
);

// Plus sign between keys
const KeyPlus: React.FC = () => (
  <Typography
    component="span"
    sx={{
      color: "#64748b",
      fontSize: "11px",
      mx: 0.5,
    }}
  >
    +
  </Typography>
);

// Render key combination
const KeyCombo: React.FC<{ keys: string[] }> = ({ keys }) => (
  <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap" }}>
    {keys.map((key, index) => (
      <React.Fragment key={index}>
        <KeyboardKey wide={key.length > 3}>{key}</KeyboardKey>
        {index < keys.length - 1 && <KeyPlus />}
      </React.Fragment>
    ))}
  </Box>
);

// Shortcut categories data
const SHORTCUT_CATEGORIES: ShortcutCategory[] = [
  {
    title: "General",
    shortcuts: [
      { keys: ["?"], description: "Show this help" },
      { keys: ["Esc"], description: "Close panels/modals" },
    ],
  },
  {
    title: "Feature Explorer",
    shortcuts: [
      { keys: ["Ctrl", "F"], description: "Search features" },
    ],
  },
  {
    title: "Video Comparison",
    shortcuts: [
      { keys: ["Space"], description: "Play/Pause" },
      { keys: ["\u2190"], description: "Previous frame" },
      { keys: ["\u2192"], description: "Next frame" },
      { keys: ["Shift", "\u2190"], description: "Jump back 5 seconds" },
      { keys: ["Shift", "\u2192"], description: "Jump forward 5 seconds" },
      { keys: ["Home"], description: "Go to start" },
      { keys: ["0"], description: "Go to start (alternate)" },
      { keys: ["End"], description: "Go to end" },
    ],
  },
];

export default function KeyboardShortcutsHelp({
  open,
  onClose,
}: KeyboardShortcutsHelpProps) {
  // Handle escape key and clicking outside
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      }
    },
    [open, onClose]
  );

  useEffect(() => {
    if (open) {
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }
  }, [open, handleKeyDown]);

  if (!open) return null;

  return (
    <Backdrop
      open={open}
      onClick={onClose}
      sx={{
        zIndex: 1300,
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        backdropFilter: "blur(4px)",
      }}
    >
      <Fade in={open}>
        <Paper
          onClick={(e) => e.stopPropagation()}
          sx={{
            position: "relative",
            maxWidth: 500,
            maxHeight: "80vh",
            overflow: "auto",
            backgroundColor: "#0f172a",
            border: "1px solid #1e3a5f",
            borderRadius: 2,
            boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
          }}
        >
          {/* Header */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              px: 3,
              py: 2,
              borderBottom: "1px solid #1e3a5f",
              backgroundColor: "#0a1628",
              position: "sticky",
              top: 0,
              zIndex: 1,
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
              <KeyboardIcon sx={{ color: "#ef4444", fontSize: 24 }} />
              <Typography
                variant="h6"
                sx={{
                  color: "white",
                  fontWeight: 600,
                  fontSize: "1rem",
                }}
              >
                Keyboard Shortcuts
              </Typography>
            </Box>
            <IconButton
              onClick={onClose}
              size="small"
              sx={{
                color: "#94a3b8",
                "&:hover": {
                  color: "white",
                  backgroundColor: "rgba(239, 68, 68, 0.2)",
                },
              }}
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>

          {/* Content */}
          <Box sx={{ p: 3 }}>
            {SHORTCUT_CATEGORIES.map((category, categoryIndex) => (
              <Box
                key={category.title}
                sx={{
                  mb: categoryIndex < SHORTCUT_CATEGORIES.length - 1 ? 3 : 0,
                }}
              >
                {/* Category Title */}
                <Typography
                  variant="overline"
                  sx={{
                    display: "block",
                    color: "#ef4444",
                    fontWeight: 700,
                    fontSize: "0.7rem",
                    letterSpacing: "0.1em",
                    mb: 1.5,
                    textTransform: "uppercase",
                  }}
                >
                  {category.title}
                </Typography>

                {/* Shortcuts List */}
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 1,
                  }}
                >
                  {category.shortcuts.map((shortcut, index) => (
                    <Box
                      key={index}
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        py: 1,
                        px: 1.5,
                        backgroundColor: "rgba(30, 41, 59, 0.5)",
                        borderRadius: 1,
                        "&:hover": {
                          backgroundColor: "rgba(30, 41, 59, 0.8)",
                        },
                      }}
                    >
                      <Typography
                        sx={{
                          color: "#cbd5e1",
                          fontSize: "0.8rem",
                          flex: 1,
                        }}
                      >
                        {shortcut.description}
                      </Typography>
                      <KeyCombo keys={shortcut.keys} />
                    </Box>
                  ))}
                </Box>
              </Box>
            ))}
          </Box>

          {/* Footer */}
          <Box
            sx={{
              px: 3,
              py: 2,
              borderTop: "1px solid #1e3a5f",
              backgroundColor: "rgba(10, 22, 40, 0.5)",
            }}
          >
            <Typography
              variant="caption"
              sx={{
                color: "#64748b",
                fontSize: "0.65rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 0.5,
              }}
            >
              Press
              <KeyboardKey>?</KeyboardKey>
              anytime to show this help
            </Typography>
          </Box>
        </Paper>
      </Fade>
    </Backdrop>
  );
}
