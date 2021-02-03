import logging
import torch.nn as nn

from selfsupmotion.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class MyModel(nn.Module):  # pragma: no cover
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """

    def __init__(self, hyper_params):
        """__init__.

        Args:
            hyper_params (dict): hyper parameters from the config file.
        """
        super(MyModel, self).__init__()

        check_and_log_hp(['size'], hyper_params)
        self.hyper_params = hyper_params
        self.linear1 = nn.Linear(5, hyper_params['size'])
        self.linear2 = nn.Linear(hyper_params['size'], 1)

    def forward(self, data):
        """Forward method of the model.

        Args:
            data (tensor): The data to be passed to the model.

        Returns:
            tensor: the output of the model computation.

        """
        hidden = nn.functional.relu(self.linear1(data))
        result = self.linear2(hidden)
        return result.squeeze()
