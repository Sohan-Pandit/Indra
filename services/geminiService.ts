
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI } from "@google/genai";
import { IndraAnnotation } from "../types";

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/annotate';

export const annotateAbstract = async (
  abstract: string,
  pubYear: string = ""
): Promise<IndraAnnotation> => {
  // Retrieve API Key
  const customKey = localStorage.getItem('INDRA_CUSTOM_KEY');
  // In production, we should handle environment variable keys safely too, 
  // but for this hybrid setup, we rely on the manual key or assume it's passed somehow.
  // FIX: Default to PLACEHOLDER_API_KEY to trigger backend Mock Mode if no key is set.
  const apiKey = customKey || "PLACEHOLDER_API_KEY";

  if (!apiKey) {
    // This should technically never happen now, but keeping for safety
    throw new Error("No API key provided.");
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey
      },
      body: JSON.stringify({
        abstract: abstract,
        pub_year: pubYear
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend Error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();

    return {
      record: data.record,
      uncertaintyAnalysis: data.uncertaintyAnalysis,
      secondaryImpacts: data.secondaryImpacts,
      rawResponse: data.rawResponse
    };

  } catch (error) {
    console.error("Annotation Service Error:", error);
    throw error;
  }
};


