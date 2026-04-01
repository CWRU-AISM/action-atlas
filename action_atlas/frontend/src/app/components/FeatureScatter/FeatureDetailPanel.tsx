import React from "react";
import { NearestFeature } from "@/types/sae";
import { AnimatePresence, motion } from "framer-motion";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { Button, Typography, Chip } from "@mui/material";
import { SimilarFeaturesPanelProps } from "@/types/types";
import { useAppDispatch, useAppSelector } from "@/redux/hooks";
import { addHistoryItem, HistoryItem } from "@/redux/features/historySlice";
import { TypeOutline } from "lucide-react";

// Parse keywords from description (handles both [keyword] format and text format)
const parseKeywords = (description: string): string[] | null => {
  // Check for bracket format: [keyword1, keyword2, ...]
  const bracketMatch = description.match(/\[([^\]]+)\]/);
  if (bracketMatch) {
    return bracketMatch[1].split(',').map(k => k.trim());
  }
  // Check for "Activates for" format: "Activates for motion: push, open; objects: bowl"
  if (description.includes('Activates for')) {
    const keywords: string[] = [];
    const parts = description.replace(/Activates for\s*/i, '').split(';');
    parts.forEach(part => {
      const colonIdx = part.indexOf(':');
      if (colonIdx > 0) {
        const items = part.slice(colonIdx + 1).split(',').map(k => k.trim());
        keywords.push(...items);
      }
    });
    return keywords.length > 0 ? keywords : null;
  }
  return null;
};

// Get chip color based on keyword type
const getKeywordColor = (keyword: string): string => {
  const upper = keyword.toUpperCase();
  // Motion/action keywords
  if (['PUT', 'PUSH', 'OPEN', 'CLOSE', 'GRASP', 'LIFT', 'PLACE', 'INTERACT', 'APPROACH'].includes(upper)) {
    return '#3b82f6'; // blue
  }
  // Object keywords
  if (['BOWL', 'PLATE', 'BOTTLE', 'DRAWER', 'CABINET', 'RACK', 'STOVE', 'CUP', 'WINE_BOTTLE'].includes(upper.replace(' ', '_'))) {
    return '#ec4899'; // pink
  }
  // Spatial keywords
  if (['ON', 'IN', 'TOP', 'FRONT', 'MIDDLE', 'ABOVE', 'BELOW'].includes(upper)) {
    return '#22c55e'; // green
  }
  // Phase keywords
  if (upper.includes('PHASE') || upper.includes('MOTION') || upper.includes('TRANSPORT')) {
    return '#f59e0b'; // amber
  }
  return '#64748b'; // gray default
};

const ToggleButton = ({
  isOpen,
  setIsOpen,
}: {
  isOpen: boolean;
  setIsOpen: React.Dispatch<React.SetStateAction<boolean>>;
}) => (
  <motion.div
    animate={{ x: isOpen ? 0 : 256 }}
    transition={{ type: "spring", stiffness: 300, damping: 30 }}
    className="absolute"
  >
    <button
      onClick={() => setIsOpen(!isOpen)}
      aria-label={isOpen ? "Hide panel" : "Show panel"}
      className="absolute top-10 right-0 w-8 h-12 bg-white shadow-md rounded-l-xl hover:bg-gray-100"
    >
      {isOpen ? <ChevronRightIcon /> : <ChevronLeftIcon />}
    </button>
  </motion.div>
);

export default function FeatureDetailPanel({
  similarFeatures,
  selectedFeatureId,
  onFeatureClick,
}: SimilarFeaturesPanelProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const dispatch = useAppDispatch();
  const selectedFeature = useAppSelector(
    (state) => state.feature.selectedFeature
  );

  return (
    <div>
      <AnimatePresence initial={false}>
        <motion.div
          initial={false}
          animate={{ x: isOpen ? 0 : 256 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="absolute right-4 top-20 z-100 flex"
        >
          <div className="w-60 max-h-[550px] rounded-r-lg rounded-bl-lg overflow-hidden flex flex-col shadow-lg bg-white">
            <div className="p-3">
              <Typography variant="subtitle1" fontWeight="bold">
                Similar to {selectedFeatureId ? selectedFeatureId : "Select"}{" "}
                (Marked:{" "}
                <span className="inline-block w-2 h-2 rounded-full bg-[#77BBDD]"></span>
                )
              </Typography>
            </div>

            <div className="flex-1 overflow-y-auto px-3 pb-3 max-h-[60vh]">
              {similarFeatures.length > 0 ? (
                <div className="space-y-2">
                  {similarFeatures.map((feat: NearestFeature) => (
                    <div key={feat.feature_id}>
                      <button
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
                              className="text-[#1976d2]"
                            >
                              ID: {feat.feature_id}
                            </Typography>
                            <Typography
                              variant="subtitle2"
                              fontWeight="bold"
                              className="text-[#1976d2]"
                            >
                              Similarity: {feat.similarity.toFixed(3)}
                            </Typography>
                          </div>
                          {(() => {
                            const keywords = parseKeywords(feat.description);
                            if (keywords) {
                              return (
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {keywords.map((kw, i) => (
                                    <Chip
                                      key={i}
                                      label={kw}
                                      size="small"
                                      sx={{
                                        height: 20,
                                        fontSize: '10px',
                                        bgcolor: `${getKeywordColor(kw)}20`,
                                        color: getKeywordColor(kw),
                                        fontWeight: 500,
                                      }}
                                    />
                                  ))}
                                </div>
                              );
                            }
                            return (
                              <p className="text-sm text-gray-600 mt-1">
                                {feat.description}
                              </p>
                            );
                          })()}
                        </div>
                      </button>
                      {/* <Button variant="contained"
                              size="small"
                              onClick={() => handleValidate()}
                              className="mt-1 mb-3 w-full">
                        Validate
                      </Button> */}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-600">
                  Click on a point to see feature details.
                </p>
              )}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="absolute top-10 right-64 z-101">
        <ToggleButton isOpen={isOpen} setIsOpen={setIsOpen} />
      </div>
    </div>
  );
}
