from linear import Linear
import numpy as np


class Binary(Linear):
    def get_potential_boundaries(self, processed_data):
        num_total = processed_data.shape[0]
        num_true = np.sum(processed_data == 1)
        yield 0
        yield num_total - num_true
        if num_true < num_total:
            yield num_total
