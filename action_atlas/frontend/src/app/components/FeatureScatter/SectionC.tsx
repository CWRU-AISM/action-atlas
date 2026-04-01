"use client";
import React from "react";
import SectionC1 from "./SectionC1";

interface FeatureInfo {
  explanation: string;
  cosine_similarity: number;
}

export default function SectionC() {
  const [selectedFeatureInfo, setSelectedFeatureInfo] = React.useState<FeatureInfo | undefined>();

  const handleFeatureSelect = (featureInfo: FeatureInfo) => {
    setSelectedFeatureInfo(featureInfo);
  };

  return <SectionC1  />;
}
