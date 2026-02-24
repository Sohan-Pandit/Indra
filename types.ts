
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

// ImpactRecord defines the structured data extracted from scientific abstracts
export interface ImpactRecord {
  abstract_id: string;
  hazard_type: 'flood' | 'drought' | 'heatwave' | 'cyclone' | 'wildfire' | 'other';
  hazard_intensity: string | null;
  location: {
    raw: string;
    normalized: string;
  };
  time_period: {
    raw: string;
    normalized: string;
  };
  impact_domain: 'mortality' | 'health' | 'agriculture' | 'infrastructure' | 'economy' | 'ecosystem' | 'displacement' | 'other';
  impact_type: 'observed' | 'projected';
  impact_magnitude: string | null;
  magnitude_vague: boolean;
  affected_group: string | null;
  causal_relation: {
    subject: string;
    predicate: 'caused' | 'contributed_to' | 'associated_with' | 'mitigated';
    object: string;
    dependency_path: string;
  };
  uncertainty_level: 'low' | 'medium' | 'high';
  uncertainty_source: 'observational' | 'modeled' | 'projected' | 'unclear';
  hedge_terms: string[];
  grounding_quote: string;
  grounding_verified: boolean;
}

// IndraAnnotation contains the extracted record and associated analysis sections
export interface IndraAnnotation {
  record: ImpactRecord;
  uncertaintyAnalysis: string;
  secondaryImpacts: string | null;
  rawResponse: string;
}

// AnnotationSession represents a single research session in the history
export interface AnnotationSession {
  id: string;
  abstract: string;
  pubYear: string;
  annotation: IndraAnnotation;
  timestamp: number;
}

/**
 * Fix: Added missing GeneratedImage interface used by the Infographic component.
 */
export interface GeneratedImage {
  id: string;
  data: string;
  prompt: string;
}

/**
 * Fix: Added missing SearchResultItem interface used by the SearchResults component.
 */
export interface SearchResultItem {
  title: string;
  url: string;
}

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}
