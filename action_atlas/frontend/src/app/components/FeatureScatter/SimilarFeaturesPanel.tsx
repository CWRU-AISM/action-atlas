import React, { useState, useEffect, useRef, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { Typography } from "@mui/material";
import { SimilarFeaturesPanelProps } from "@/types/types";

const ToggleButton = ({
  isOpen,
  setIsOpen,
}: {
  isOpen: boolean;
  setIsOpen: React.Dispatch<React.SetStateAction<boolean>>;
}) => (
  <motion.div
    animate={{ x: isOpen ? 0 : -256 }}
    transition={{ type: "spring", stiffness: 300, damping: 30 }}
    className="absolute"
  >
    <button
      onClick={() => setIsOpen(!isOpen)}
      aria-label={isOpen ? "Hide panel" : "Show panel"}
      className="absolute top-10 left-64 w-8 h-12 bg-white shadow-md rounded-r-lg hover:bg-gray-100"
    >
      {isOpen ? <ChevronLeftIcon /> : <ChevronRightIcon />}
    </button>
  </motion.div>
);

export default function SimilarFeaturesPanel({
  similarFeatures,
  selectedFeatureId,
  hasQueryPoint,
  onFeatureClick,
  onVisibleFeaturesChange,
}: SimilarFeaturesPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const featureRefsMap = useRef(new Map<string, HTMLButtonElement | null>());
  const observerRef = useRef<IntersectionObserver | null>(null);

  useCallback((id: string, element: HTMLButtonElement | null) => {
    if (featureRefsMap.current) {
      featureRefsMap.current.set(id, element);
    }
  }, []);
  useEffect(() => {
    if (!onVisibleFeaturesChange || similarFeatures.length === 0) return;

    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const visibleIds = entries
          .filter((entry) => entry.isIntersecting)
          .map((entry) => entry.target.getAttribute("data-feature-id"))
          .filter(Boolean) as string[];

        onVisibleFeaturesChange(visibleIds);
      },
      { threshold: 0.5 }
    );

    // Get all non-null refs from the Map
    const currentFeatureRefs: HTMLButtonElement[] = [];
    featureRefsMap.current.forEach((ref) => {
      if (ref) currentFeatureRefs.push(ref);
    });

    currentFeatureRefs.forEach((ref) => {
      if (ref) observerRef.current?.observe(ref);
    });

    return () => {
      observerRef.current?.disconnect();
    };
  }, [similarFeatures, onVisibleFeaturesChange]);

  return (
    <div>
      <AnimatePresence initial={false}>
        <motion.div
          initial={false}
          animate={{ x: isOpen ? 0 : -256 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="absolute left-4 top-20 z-100 flex"
        >
          <div className="w-60 max-h-[550px] rounded-l-lg rounded-br-lg overflow-hidden flex flex-col shadow-lg bg-white">
            <div className="p-3">
              <Typography variant="subtitle1" fontWeight="bold">
                Similar to Query (Marked:{" "}
                <span className="inline-block w-2 h-2 rounded-full bg-[#FF8899]"></span>
                )
              </Typography>
            </div>

            <div className="flex-1 overflow-y-auto px-3 pb-3 max-h-[60vh]">
              {hasQueryPoint ? (
                similarFeatures.length > 0 ? (
                  <div className="space-y-2">
                    {similarFeatures.map((feat) => (
                      <button
                        key={feat.feature_id}
                        ref={(el) => {
                          if (featureRefsMap.current) {
                            featureRefsMap.current.set(feat.feature_id, el);
                          }
                        }}
                        data-feature-id={feat.feature_id}
                        onClick={() => onFeatureClick(feat)}
                        className={`w-full text-left p-2 rounded-md mb-2 border transition-colors duration-300 ${
                          selectedFeatureId === feat.feature_id
                            ? "bg-blue-100"
                            : "bg-white"
                        }`}
                      >
                        <div className="flex flex-col">
                          <div className="flex items-center justify-between">
                            <Typography
                              variant="subtitle2"
                              fontWeight="bold"
                              className="text-[#1976d1]"
                            >
                              ID: {feat.feature_id}
                            </Typography>
                            <Typography
                              variant="subtitle2"
                              fontWeight="bold"
                              className="text-[#1976d1]"
                            >
                              Similarity: {feat.similarity.toFixed(3)}
                            </Typography>
                          </div>
                          <Typography
                            variant="body2"
                            className="text-gray-500 mt-1"
                          >
                            {feat.description}
                          </Typography>
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-600">
                    No similar features found.
                  </p>
                )
              ) : (
                <p className="text-sm text-gray-600">
                  Search for features to see similarities.
                </p>
              )}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="absolute top-10 z-101">
        <ToggleButton isOpen={isOpen} setIsOpen={setIsOpen} />
      </div>
    </div>
  );
}
