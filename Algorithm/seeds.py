"""
Centralized seed management to ensure consistency and avoid overlaps.

DATASET COLLECTION (already collected - DO NOT MODIFY):
- starting_seed = 2000
- num_envs = 50
- Seeds used in the dataset: from 2000 onward, organized as:
  Row 0: 2000, 2001, ..., 2049
  Row 1: 2050, 2051, ..., 2099
  ...
  
For 100k samples with 50 envs -> approximately 2000 rows -> seeds from 2000 to ~102000
For safety we consider used: 2000 - 150000

USAGE:
- Scripts in Algorithm/: can import the constants directly (SEEDS_ID, TRAINING_SEED, etc.)
- External scripts (Test/, etc.): should use the getter functions (get_seeds_id(), get_training_seed(), etc.)
"""

# ============================================================================
# PRIVATE CONSTANTS (use only within Algorithm/)
# ============================================================================
_DATASET_STARTING_SEED = 2000
_DATASET_SEED_RANGE = (2000, 150_000)

_SEEDS_ID = list(range(2000, 2010))
_SEEDS_OD = list(range(1900, 1910))

_TRAINING_SEED = 600_000
_PHASE_ONE_SEED_BASE = 700_000
_PHASE_TWO_ENV_RESET_SEED = 800_000
_PHASE_TWO_SEED_BASE = 800_000

_MODEL_SEEDS = [0, 1, 2, 3, 4, 5]  # Seeds for model weight initialization

_SEED_OFFSET = 10_000  # Each run has a 10k seed margin

_DATALOADER_SEED = [10, 11, 12, 13, 14]

# ============================================================================
# PUBLIC ALIASES (for internal use within Algorithm/)
# ============================================================================
DATASET_STARTING_SEED = _DATASET_STARTING_SEED
DATASET_SEED_RANGE = _DATASET_SEED_RANGE
SEEDS_ID = _SEEDS_ID
SEEDS_OD = _SEEDS_OD
TRAINING_SEED = _TRAINING_SEED
PHASE_ONE_SEED_BASE = _PHASE_ONE_SEED_BASE
PHASE_TWO_ENV_RESET_SEED = _PHASE_TWO_ENV_RESET_SEED
PHASE_TWO_SEED_BASE = _PHASE_TWO_SEED_BASE
MODEL_SEEDS = _MODEL_SEEDS
SEED_OFFSET = _SEED_OFFSET
DATALOADER_SEED = _DATALOADER_SEED

# ============================================================================
# GETTER FUNCTIONS (for external use - Test/, etc.)
# ============================================================================
def get_seeds_id() -> list:
    """Return the In-Distribution seeds (used in the dataset)."""
    return _SEEDS_ID.copy()

def get_seeds_od() -> list:
    """Return the Out-of-Distribution seeds (NOT used in the dataset)."""
    return _SEEDS_OD.copy()

def get_dataset_starting_seed() -> int:
    """Return the starting seed used to collect the dataset."""
    return _DATASET_STARTING_SEED

def get_training_seed(run_index: int = 0) -> int:
    """Return the training seed for a specific run."""
    return _TRAINING_SEED + run_index * _SEED_OFFSET

def get_phase_one_seed_base(run_index: int = 0) -> int:
    """Return the base seed for Phase One of a specific run."""
    return _PHASE_ONE_SEED_BASE + run_index * _SEED_OFFSET

def get_phase_two_seeds(run_index: int = 0) -> tuple:
    """Return (env_reset_seed, seed_base) for Phase Two of a specific run."""
    offset = run_index * _SEED_OFFSET
    return (_PHASE_TWO_ENV_RESET_SEED + offset, _PHASE_TWO_SEED_BASE + offset)

def get_model_seed(run_index: int = 0) -> int:
    """Return the seed for model weight initialization."""
    return _MODEL_SEEDS[run_index % len(_MODEL_SEEDS)]

def is_seed_in_dataset(seed: int) -> bool:
    """Check whether a seed was used in dataset collection."""
    return _DATASET_SEED_RANGE[0] <= seed <= _DATASET_SEED_RANGE[1]

def get_dataloader_seed(run_index: int = 0) -> int:
    """Return a seed for the DataLoader, based on the run index."""
    return _DATALOADER_SEED[run_index % len(_DATALOADER_SEED)]
