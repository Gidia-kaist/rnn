import torch


class SharedPreference:

    # Default setting
    filter_mask = False
    error_mask = False
    boolean_mask = torch.ones(1600)
    boolean_mask = boolean_mask.cuda()

    count_inpts_exc = 0
    count_exc_inh = 0
    count_inh_exc = 0

    copy_w = torch.zeros(784, 1600)

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
        self.boolean_mask[num] = disc
        return self.boolean_mask

    # Load the connectivity data
    def get_boolean_mask(self):
        return self.boolean_mask

    def set_connection_w(self, row: int, col: int, disc: int):
        self.input_exc_conn.w[row, col] = disc
        return self.input_exc_conn.w

    def get_connection_w(self):
        return self.input_exc_conn.w

    def initialize_count(self):
        self.count_inpts_exc = 0
        self.count_exc_inh = 0
        self.count_inh_exc = 0

    def set_count(self, code: int):
        if code == 1:
            self.count_inpts_exc += 1
            return self.count_inpts_exc
        elif code == 2:
            self.count_exc_inh += 1
            return self.count_exc_inh
        elif code == 3:
            self.count_inh_exc += 1
            return self.count_inh_exc

    def get_count(self, code: int):
        if code == 1:
            return self.count_inpts_exc
        elif code == 2:
            return self.count_exc_inh
        elif code == 3:
            return self.count_inh_exc

    def set_copy(self, target: torch.tensor, col: int):
        self.copy_w[:, col] = target[:, col]
        return self.copy_w

    def get_copy(self):
        return self.copy_w

    def set_error_on(self, boolean_inpt: bool):
        self.error_mask = boolean_inpt
        return self.error_mask

    def get_error(self):
        return self.error_mask
