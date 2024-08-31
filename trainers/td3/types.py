import dataclasses as dc
from networks import Network
from torch.optim.optimizer import Optimizer
import torch as t
from typing_extensions import TypeVar, Generic

NetworkVar = TypeVar("NetworkVar", bound=Network, covariant=True)
OptimizerVar = TypeVar(
    "OptimizerVar", bound=Optimizer, covariant=True
)


@dc.dataclass
class NetworkSet(Generic[NetworkVar, OptimizerVar]):
    learning: NetworkVar
    target: NetworkVar
    optimizer: OptimizerVar
