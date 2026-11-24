"""
Flat vector view of numerical parameters across models.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import logging
from collections import OrderedDict

from andes.core.param import ExtParam, NumParam
from andes.shared import np, pd

logger = logging.getLogger(__name__)


class ParamVector:
    """
    A flat, ordered vector of selected numerical parameters.

    ``ParamVector`` maps the parameters of many models to one 1-D array, as
    used by optimizers, sensitivity analysis and parameter estimation. Values
    are read from and written to the parameter arrays of the models
    (``Model.<param>.v``), so the vector is not a copy of the data: the
    models remain the single source of truth.

    Use :meth:`all` for a vector of every parameter in the system, and
    :meth:`subset` to narrow a vector down.

    After writing new values, call ``TDS.reinit(keep_params=True)`` (dynamic
    models) or run the power flow again (power flow models) for the change to
    take effect; see :attr:`pflow` for which entries need which.

    Parameters
    ----------
    system : System
        A system on which ``setup()`` has been called.
    select : list
        Parameters to include, in order. Each item is one of

        - ``'Model.param'`` or ``('Model', 'param')``: all devices of the model;
        - ``('Model', 'param', idx)``: the devices with the given ``idx``,
          a single value or a list.

        Each ``(model, param)`` pair may appear only once.
    base : None or 'device', optional
        ``None`` (default) reads and writes values in system-base per unit,
        as stored in ``v`` and used in the equations. ``'device'`` uses the
        device base, as in the case file.

    Attributes
    ----------
    names : list of str
        Entry names in the form ``'Model.param[idx]'``.
    model, param : list of str
        Model and parameter name of each entry.
    idx : list
        Device ``idx`` of each entry.
    pflow : np.ndarray of bool
        True for entries that belong to power flow models (such as ``Line``
        or ``PQ``), whose changes require a new power flow.

    Examples
    --------
    >>> ss = andes.load(andes.get_case('kundur/kundur_full.xlsx'))
    >>> ss.PFlow.run()
    >>> ss.TDS.init()
    >>> pv = andes.core.ParamVector(ss, ['GENROU.M', ('TGOV1', 'R', [1, 2])])
    >>> p = pv.get()
    >>> pv.set(1.1 * p)
    >>> ss.TDS.reinit(keep_params=True)
    >>> ss.TDS.run()
    """

    def __init__(self, system, select, base=None):
        if not system.is_setup:
            raise RuntimeError("ParamVector requires a system after setup().")
        if base not in (None, 'device'):
            raise ValueError(f"base must be None (system base) or 'device', got {base!r}")

        self.system = system
        self.base = base

        # one block per (model, param): (model, NumParam, uids as int array, slice)
        self._blocks = OrderedDict()
        self.names, self.model, self.param, self.idx = [], [], [], []
        pflow = []

        offset = 0
        for item in select:
            mname, pname, idx = self._parse(item)
            mdl, instance = self._lookup(mname, pname)

            if (mname, pname) in self._blocks:
                raise ValueError(f"{mname}.{pname} is selected more than once.")

            if idx is None:
                idx = list(mdl.idx.v)
            elif isinstance(idx, (str, int, float, np.integer, np.floating)):
                idx = [idx]
            else:
                idx = list(idx)

            uid = np.array(mdl.idx2uid(idx), dtype=int)
            if len(set(uid.tolist())) != len(uid):
                raise ValueError(f"{mname}.{pname}: duplicate idx in {idx}.")

            n = len(uid)
            self._blocks[(mname, pname)] = (mdl, instance, uid, slice(offset, offset + n))
            offset += n

            self.names.extend(f'{mname}.{pname}[{i}]' for i in idx)
            self.model.extend([mname] * n)
            self.param.extend([pname] * n)
            self.idx.extend(idx)
            pflow.extend([mname in system.exist.pflow] * n)

        self.size = offset
        self.pflow = np.array(pflow, dtype=bool)

    @staticmethod
    def _parse(item):
        """Parse one selection item into ``(model, param, idx)``."""
        if isinstance(item, str):
            parts = item.split('.')
            if len(parts) != 2:
                raise ValueError(f"Expected 'Model.param', got {item!r}.")
            return parts[0], parts[1], None

        if isinstance(item, (tuple, list)) and len(item) in (2, 3):
            idx = item[2] if len(item) == 3 else None
            return item[0], item[1], idx

        raise ValueError(f"Cannot parse selection item {item!r}; use 'Model.param', "
                         "('Model', 'param') or ('Model', 'param', idx).")

    def _lookup(self, mname, pname):
        """Return the model and the parameter instance, checking that it can be used."""
        mdl = self.system.models.get(mname)
        if mdl is None:
            raise KeyError(f"<{mname}> is not a model.")

        instance = mdl.__dict__.get(pname)
        reason = self._unusable(mdl, instance)
        if reason is not None:
            raise ValueError(f"{mname}.{pname} {reason}.")

        return mdl, instance

    @staticmethod
    def _unusable(mdl, instance):
        """Return why ``instance`` cannot be in the vector, or None if it can."""
        if not isinstance(instance, NumParam):
            return "is not a numerical parameter (NumParam)"
        if isinstance(instance, ExtParam):
            return (f"is an ExtParam, a copy of {instance.model}.{instance.src}; "
                    "select the source instead")
        if instance is mdl.u or instance is getattr(mdl, 'ue', None):
            return "is the device status; use System.set_status() instead"
        if not isinstance(instance.v, np.ndarray) or instance.v.dtype.kind != 'f' \
                or instance.v.ndim != 1:
            return "does not hold one float per device"
        return None

    @classmethod
    def all(cls, system, models=None, pflow=True, base=None):
        """
        Build a vector of every numerical parameter of every device.

        Entries are ordered by model (in ``System.models`` order), then
        parameter (in definition order), then device. Device status (``u``),
        ``ExtParam`` copies and non-float parameters are left out, as they
        cannot be set through the vector.

        Parameters
        ----------
        system : System
            A system on which ``setup()`` has been called.
        models : list of str, optional
            Only include these models. Default: all models with devices.
        pflow : bool, optional
            ``False`` leaves out power flow models (``Bus``, ``Line``,
            ``PQ``, ...), leaving the parameters that
            ``TDS.reinit(keep_params=True)`` can apply. Default ``True``.
        base : None or 'device', optional
            As in :class:`ParamVector`.

        Returns
        -------
        ParamVector

        Examples
        --------
        >>> pv = ParamVector.all(ss, pflow=False)
        >>> pv.as_df().query("model == 'TGOV1'")
        """
        if not system.is_setup:
            raise RuntimeError("ParamVector requires a system after setup().")
        if models is not None:
            unknown = [m for m in models if m not in system.models]
            if unknown:
                raise KeyError(f"Not models: {', '.join(unknown)}.")

        select = []
        for mname, mdl in system.models.items():
            if mdl.n == 0 or (models is not None and mname not in models):
                continue
            if not pflow and mname in system.exist.pflow:
                continue
            for pname, instance in mdl.num_params.items():
                if cls._unusable(mdl, instance) is None:
                    select.append((mname, pname))

        return cls(system, select, base=base)

    def subset(self, mask):
        """
        Return a new ``ParamVector`` with the selected entries.

        Entries keep their order within each ``(model, param)``; parameters
        appear in the order of their first selected entry.

        Parameters
        ----------
        mask : array-like of bool or int
            A boolean mask of length :attr:`size`, or entry positions.

        Returns
        -------
        ParamVector

        Examples
        --------
        >>> pv = ParamVector.all(ss)
        >>> dyn = pv.subset(~pv.pflow)
        >>> gains = pv.subset([pv.index('TGOV1', 'R', 1), pv.index('EXDC2', 'KA', 1)])
        """
        mask = np.asarray(mask)
        if mask.dtype == bool:
            if mask.shape != (self.size,):
                raise ValueError(f"Boolean mask must have length {self.size}.")
            positions = np.flatnonzero(mask)
        else:
            positions = mask.astype(int).ravel()
            if positions.size and (positions.min() < -self.size or positions.max() >= self.size):
                raise IndexError("Entry position out of range.")
            positions = positions % self.size if self.size else positions

        grouped = OrderedDict()
        for i in positions.tolist():
            grouped.setdefault((self.model[i], self.param[i]), []).append(self.idx[i])

        select = [(m, p, idx) for (m, p), idx in grouped.items()]
        return ParamVector(self.system, select, base=self.base)

    def __len__(self):
        return self.size

    def __repr__(self):
        return f'ParamVector ({self.size} entries, {len(self._blocks)} parameters)'

    def get(self):
        """
        Return the current values as a new 1-D array.

        Returns
        -------
        np.ndarray
            Values in the order of :attr:`names`.
        """
        out = np.empty(self.size)
        for mdl, instance, uid, sl in self._blocks.values():
            values = instance.v[uid]
            if self.base == 'device':
                values = values / instance.pu_coeff[uid]
            out[sl] = values
        return out

    def set(self, values):
        """
        Write values to the model parameters, in place.

        Time constants of differential equations (such as ``GENROU.M``) are
        also updated in the DAE mass matrix, as in :meth:`Model.set`.

        Parameters
        ----------
        values : array-like
            New values in the order of :attr:`names`, of length :attr:`size`.
        """
        values = np.asarray(values, dtype=float).ravel()
        if values.shape != (self.size,):
            raise ValueError(f"Expected {self.size} values, got {values.size}.")

        for mdl, instance, uid, sl in self._blocks.values():
            block = values[sl]
            if self.base == 'device':
                if instance.vin is not None:
                    instance.vin[uid] = block
                block = block * instance.pu_coeff[uid]
            instance.v[uid] = block
            mdl._update_time_constants(instance, uid.tolist())

    def index(self, model, param, idx=None):
        """
        Return the positions of entries in the vector.

        Parameters
        ----------
        model : str
            Model name.
        param : str
            Parameter name.
        idx : optional
            Device idx, a single value or a list. ``None`` returns all
            entries of the parameter.

        Returns
        -------
        int or np.ndarray
            One position for a single ``idx``, otherwise an integer array.
        """
        if (model, param) not in self._blocks:
            raise KeyError(f"{model}.{param} is not in the vector.")

        mdl, _, uid, sl = self._blocks[(model, param)]
        positions = np.arange(sl.start, sl.stop)
        if idx is None:
            return positions

        single = isinstance(idx, (str, int, float, np.integer, np.floating))
        wanted = mdl.idx2uid([idx] if single else list(idx))
        lookup = {u: p for u, p in zip(uid.tolist(), positions.tolist())}
        try:
            out = np.array([lookup[u] for u in wanted], dtype=int)
        except KeyError:
            raise KeyError(f"{model}.{param}: idx {idx} is not in the vector.") from None
        return int(out[0]) if single else out

    def as_df(self):
        """
        Return a DataFrame with one row per entry: model, parameter, idx,
        value (in the vector's base), unit, whether it is a power flow
        parameter, and its description.
        """
        units = []
        info = []
        for mdl, instance, uid, sl in self._blocks.values():
            units.extend([instance.unit] * len(uid))
            info.extend([instance.info] * len(uid))

        return pd.DataFrame({'model': self.model,
                             'param': self.param,
                             'idx': self.idx,
                             'value': self.get(),
                             'unit': units,
                             'pflow': self.pflow,
                             'info': info,
                             }, index=self.names)
