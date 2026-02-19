export type Stage = "orientation" | "architecture" | "implementation" | "ambiguity" | "training";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type HyperparameterEntry = {
  name: string;
  value?: string | null;
  source_section: string;
  status: "paper-stated" | "inferred" | "assumed" | "missing";
  suggested_default?: string | null;
};

export type AmbiguityEntry = {
  ambiguity_id: string;
  ambiguous_point: string;
  section: string;
  implementation_consequence: string;
  best_guess_resolution: string;
  reasoning: string;
  resolved: boolean;
  user_resolution?: string | null;
};

export type ConversationState = {
  session_id: string;
  current_stage: Stage;
  message_history: ChatMessage[];
  resolved_ambiguities: Record<string, string>;
  current_component_index?: number;
  user_level?: string;
  pending_question?: string | null;
  last_component_focus?: string | null;
  metadata?: Record<string, unknown>;
  internal_representation: {
    problem_statement: string;
    method_summary: string;
    novelty: string;
    component_graph: { parent: string; child: string }[];
    hyperparameter_registry: HyperparameterEntry[];
    ambiguity_log: AmbiguityEntry[];
    training_procedure: string;
    prerequisite_concepts: { concept: string; explanation: string }[];
  };
};

export type ArtifactItem = {
  kind: string;
  filename: string;
  content_type: string;
  content: string;
};
