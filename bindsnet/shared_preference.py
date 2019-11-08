import torch


class SharedPreference:

    # Default setting
    filter_mask = False
    boolean_mask = torch.ones(784, 100)
    boolean_mask = boolean_mask.cuda()
    def __init__(self):
        self.filter_mask = False

    # Check the accuracy for making decision whether set the connectivity or not
    def set_filter_mask(self, boolean_inpt: bool):
        self.filter_mask = boolean_inpt
        return self.filter_mask

    # Load the boolean data
    def get_filter_mask(self):
        return self.filter_mask

    # Set the connectivity
    def set_boolean_mask(self, num: int, disc: int):
        self.boolean_mask[:, num] = disc
        return self.boolean_mask

    # Load the connectivity data
    def get_boolean_mask(self):
        return self.boolean_mask

    def set_connection_w(self, row: int, col: int, disc: int):
        self.input_exc_conn.w[row, col] = disc
        return self.input_exc_conn.w

    def get_connection_w(self):
        return self.input_exc_conn.w