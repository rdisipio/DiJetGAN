import numpy as np


class FourMomentumScaler(object):

    def __init__(self, p4repr="PtEtaPhiM"):
        self.p4repr = p4repr

        self._max_pt = 1000.
        self._max_pt_jet = 1000.
        self._max_eta = 5.
        self._max_eta_jet = 2.5
        self._max_phi = np.pi
        self._max_M = 5000.
        self._max_M_jet = 300.
        self._max_px = 2000.
        self._max_py = 2000.
        self._max_pz = 3000.
        self._max_E = 5000.
        self._max_dPhi = np.pi
        self._max_dEta = 5.
        self._max_dR = 10.

    def transform(self, X):
        n = int(len(X[0, :]) / 4)

        if self.p4repr == "PtEtaPhiM":
            for i in range(n):
                k = 4*i
                X[:, k+0] /= self._max_pt
                X[:, k+1] /= self._max_eta
                X[:, k+2] /= self._max_phi
                X[:, k+3] /= self._max_M
        elif self.p4repr == "PxPyPzE":
            for i in range(n):
                k = 4*i
                X[:, k+0] /= self._max_px
                X[:, k+1] /= self._max_py
                X[:, k+2] /= self._max_pz
                X[:, k+3] /= self._max_E

        elif self.p4repr == "PtEtaPhiEM":
            # j1
            X[:, 0] /= self._max_pt_jet
            X[:, 1] /= self._max_eta_jet
            X[:, 2] /= self._max_phi
            X[:, 3] /= self._max_E
            X[:, 4] /= self._max_M_jet
            # j2
            X[:, 5] /= self._max_pt_jet
            X[:, 6] /= self._max_eta_jet
            X[:, 7] /= self._max_phi
            X[:, 8] /= self._max_E
            X[:, 9] /= self._max_M_jet

        elif self.p4repr == "PtEtaPhiEMdR":
            # j1
            X[:, 0] /= self._max_pt_jet
            X[:, 1] /= self._max_eta
            X[:, 2] /= self._max_phi
            X[:, 3] /= self._max_E
            X[:, 4] /= self._max_M_jet
            # j2
            X[:, 5] /= self._max_pt_jet
            X[:, 6] /= self._max_eta
            X[:, 7] /= self._max_phi
            X[:, 8] /= self._max_E
            X[:, 9] /= self._max_M_jet
            # jj
            X[:, 10] /= self._max_pt
            X[:, 11] /= self._max_eta
            X[:, 12] /= self._max_phi
            X[:, 13] /= self._max_E
            X[:, 14] /= self._max_M
            # angles
            X[:, 15] /= self._max_dPhi
            X[:, 16] /= self._max_dEta
            X[:, 17] /= self._max_dR

    def inverse_transform(self, X):
        n = int(len(X[0, :]) / 4)

        if self.p4repr == "PtEtaPhiM":
            for i in range(n):
                k = 4*i
                X[:, k+0] *= self._max_pt
                X[:, k+1] *= self._max_eta
                X[:, k+2] *= self._max_phi
                X[:, k+3] *= self._max_M
        elif self.p4repr == "PxPyPzE":
            for i in range(n):
                k = 4*i
                X[:, k+0] *= self._max_px
                X[:, k+1] *= self._max_py
                X[:, k+2] *= self._max_pz
                X[:, k+3] *= self._max_E

        elif self.p4repr == "PtEtaPhiEM":
            # j1
            X[:, 0] *= self._max_pt_jet
            X[:, 1] *= self._max_eta_jet
            X[:, 2] *= self._max_phi
            X[:, 3] *= self._max_E
            X[:, 4] *= self._max_M_jet
            # j2
            X[:, 5] *= self._max_pt_jet
            X[:, 6] *= self._max_eta_jet
            X[:, 7] *= self._max_phi
            X[:, 8] *= self._max_E
            X[:, 9] *= self._max_M_jet

        elif self.p4repr == "PtEtaPhiEMdR":
            # j1
            X[:, 0] *= self._max_pt_jet
            X[:, 1] *= self._max_eta
            X[:, 2] *= self._max_phi
            X[:, 3] *= self._max_E
            X[:, 4] *= self._max_M_jet
            # j2
            X[:, 5] *= self._max_pt_jet
            X[:, 6] *= self._max_eta
            X[:, 7] *= self._max_phi
            X[:, 8] *= self._max_E
            X[:, 9] *= self._max_M_jet
            # jj
            X[:, 10] *= self._max_pt
            X[:, 11] *= self._max_eta
            X[:, 12] *= self._max_phi
            X[:, 13] *= self._max_E
            X[:, 14] *= self._max_M
            # angles
            X[:, 15] *= self._max_dPhi
            X[:, 16] *= self._max_dEta
            X[:, 17] *= self._max_dR
