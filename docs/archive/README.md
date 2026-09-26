# Archive

These documents describe the first version of the project: 494 cases filled
from 15 templates and a keyword-matching scorer. That approach was replaced.
An audit found that the templates leaked the answer in the scenario list, that
the train and test splits shared templates, and that the keyword scorer ranked
a keyword dump above careful answers.

They are kept for history. Claims in them, such as a 100% validation pass rate,
refer to the old system and are not current. See the main README for the
current design and measured results.
