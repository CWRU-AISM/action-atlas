"use client";
import React from "react";
import { Box, Button, Skeleton, Typography } from "@mui/material";
import { FeatureSectionProps } from "@/types/types";
import { useAppSelector } from "@/redux/hooks";
import { setValidateFeatureId } from "@/redux/features/featureSlice";
import { useDispatch } from "react-redux";

const FeatureExplanation = ({ featureInfo }: FeatureSectionProps) => {
  const selectedFeature = useAppSelector(
    (state) => state.feature.selectedFeature
  );
  const displayInfo = featureInfo || selectedFeature;
  const dispatch = useDispatch();
  const isLoading = useAppSelector((state) => state.feature.isLoading);

  const handleValidate = () => {
    console.log(displayInfo);
    if (displayInfo?.data?.feature_info?.feature_id) {
      dispatch(setValidateFeatureId(displayInfo.data.feature_info.feature_id));
    }
  };

  return (
    <Box className="border-b-2 border-gray-200 w-full">
      <div className="flex justify-between items-start">
        <div className="flex flex-col gap-1">
          <Typography
            variant="subtitle1"
            fontWeight="bold"
            sx={{ fontSize: "20px" }}
          >
            Feature Details{" "}
            {!isLoading && displayInfo?.data.feature_info.feature_id
              ? `(ID: ${displayInfo.data.feature_info.feature_id})`
              : ""}
          </Typography>

          {isLoading ? (
            <div>
              <Skeleton variant="text" width={300} height={24} />
              <Skeleton variant="rectangular" width={400} height={60} />
              <Skeleton variant="text" width={200} height={20} />
            </div>
          ) : (
            <div>
              <Typography
                variant="body2"
                className="text-gray-600 leading-relaxed"
                fontSize="1.3rem"
              >
                {displayInfo?.data.explanation || "No explanation available"}
              </Typography>
            </div>
          )}
        </div>

        {isLoading ? (
          <Skeleton variant="rectangular" width={100} height={36} />
        ) : (
          <Button
            variant="contained"
            className="bg-blue-600 text-white my-auto w-[100px]"
            onClick={handleValidate}
          >
            Validate
          </Button>
        )}
      </div>
    </Box>
  );
};

export default FeatureExplanation;
