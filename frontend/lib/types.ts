export type BriefingSectionKey =
  | "section_1"
  | "section_2"
  | "section_3"
  | "section_4"
  | "section_5"
  | "section_6";

export type HyperparameterEntry = {
  name: string;
  value?: string | null;
  source_section: string;
  status: "paper-stated" | "inferred" | "missing" | "assumed";
  suggested_default?: string | null;
  suggested_reasoning?: string | null;
};

export type AmbiguityEntry = {
  ambiguity_id: string;
  ambiguity_type:
  | "missing_hyperparameter"
  | "undefined_notation"
  | "underspecified_architecture"
  | "missing_training_detail"
  | "ambiguous_loss_function";
  title: string;
  ambiguous_point: string;
  section: string;
  implementation_consequence: string;
  agent_resolution: string;
  reasoning: string;
  confidence: number;
  resolved: boolean;
  user_resolution?: string | null;
};

export type PrerequisiteEntry = {
  concept: string;
  problem: string;
  solution: string;
  usage_in_paper: string;
};

export type ConversationState = {
  paper_id: string;
  paper_title?: string;
  status: "PROCESSING" | "COMPLETE" | "FAILED";
  sections: Record<BriefingSectionKey, boolean>;
  briefing: Record<BriefingSectionKey, string | null>;
  hyperparameters: HyperparameterEntry[];
  ambiguities: AmbiguityEntry[];
  prerequisites: PrerequisiteEntry[];
  resolved_ambiguities: Record<string, string>;
  code_snippets?: any[];
};

export type ArtifactItem = {
  kind: string;
  filename: string;
  content_type: string;
  content: string;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type AuthProfile = {
  id: string;
  email: string;
  name: string;
  avatar_url?: string | null;
  plan: "FREE" | "PRO";
  papers_analyzed: number;
  limit: number | "unlimited";
};
