"""The shape every tool has: a name, a one-line description for the LLM, and a run() method."""

from ..schemas import ModalityType


class BaseClinicalTool:
    name = "base_tool"
    description = "Base clinical tool"
    modality = ModalityType.CLINICAL_NOTES

    def run(self, case, **options):
        """Looks at the patient case and returns a ToolResult. Every real tool replaces this."""
        raise NotImplementedError
