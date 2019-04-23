import numpy as np


class Tensor:
    """
    A container class for a tensor representing either a state vector
    or an operator, with methods for common functionality of both
    :py:class:`.State` and :py:class:`.Operator`.

    Parameters
    ----------
    tensor : numpy complex128 multidimensional array
        The tensor representing either a quantum state or an operator
        on the vector space for a quantum state.
    """

    def __init__(self, tensor):
        self._t = np.array(tensor, dtype=np.complex128)

    def __str__(self):
        return str(self._t)

    def __getitem__(self, key):
        return self._t[key]

    # TODO include the setter?
    #def __setitem__(self, key, value):
    #    self._t[key] = value

    @property
    def shape(self):
        """
        Get the shape of the tensor. For `d` qubit systems, a
        :py:class:`.State` will have shape [2] :math:`\\times d`,
        while an :py:class:`.Operator` will have shape
        [2] :math:`\\times 2d`.

        Returns
        -------
        int
            The tensor shape.
        """
        return self._t.shape

    @property
    def rank(self):
        """
        Get the rank of the tensor. For `d` qubit systems, a
        :py:class:`.State` will have rank `d`, while an
        :py:class:`.Operator` with have rank `2d`

        Returns
        -------
        int
            The tensor rank.
        """

        return len(self.shape)

    def tensor_product(self, arg):
        """
        Return the :py:class:`.State` or :py:class:`.Operator`
        given by the tensor product of this object with another
        :py:class:`.State` or :py:class:`.Operator`. Can also be
        called with the infix `*` operator, i.e., A * B.

        Parameters
        ----------
        arg : State
            The :py:class:`.State` or :py:class:`.Operator` with which
            to take the tensor product.

        Returns
        -------
        State or Operator (depends on argument type)
            If this object has tensor `A`, and argument has tensor `B`,
            returns :math:`A\\otimes B`.
        """

        return self.__class__(np.tensordot(self._t, arg._t, axes=0))

    def tensor_power(self, n):
        """
        Return the :py:class:`.State` or :py:class:`.Operator`
        given by the tensor product of this object with itself `n`
        times. Can also be called with the infix ** operator, i.e.,
        A**n.

        Parameters
        ----------
        n : int
            The number of times to take the tensor product of this
            object with itself.

        Returns
        -------
        State or Operator (depends on this object type)
            If this object has tensor `A`,
            returns :math:`A^{\\otimes n}`.
        """
        t = self._t
        for i in range(n-1):
            t = np.tensordot(t, self._t, axes=0)

        return self.__class__(t)

    def __mul__(self, arg):
        return self.tensor_product(arg)

    def __pow__(self, n):
        return self.tensor_power(n)
