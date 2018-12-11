import util as U_
import numpy as np

class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class UR_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, randomize=True):
        traj_data = np.load(expert_path)
        obs = traj_data['obs']
        acs = traj_data['acs']

        self.rets = np.zeros_like(obs)

        self.obs = obs
        self.acs = acs
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)

        self.train_set = Dset(self.obs[:int(len(self.obs)*train_fraction), :],
                              self.acs[:int(len(self.obs)*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(len(self.obs)*train_fraction):, :],
                            self.acs[int(len(self.obs)*train_fraction):, :],
                            self.randomize)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def get_ob_shape(self):
        return self.obs.shape[1]

    def get_ac_shape(self):
        return self.acs.shape[1]

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()

def test(expert_path, plot):
    dset = UR_Dset(expert_path)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default= U_.getDataPath() + "/obs_acs.npz")
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.plot)
