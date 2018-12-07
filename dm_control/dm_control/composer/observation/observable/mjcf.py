# Copyright 2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Observables that are defined in terms of MJCF elements."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mjcf
from dm_control.composer.observation.observable import base
import numpy as np
from dm_control.rl import specs


def _check_mjcf_element(obj):
  if not isinstance(obj, mjcf.Element):
    raise ValueError(
        'expected an `mjcf.Element`, got type {}: {}'.format(type(obj), obj))


def _check_mjcf_element_iterable(obj_iterable):
  if not isinstance(obj_iterable, collections.Iterable):
    obj_iterable = (obj_iterable,)
  for obj in obj_iterable:
    _check_mjcf_element(obj)


class MJCFFeature(base.Observable):
  """An observable corresponding to an element in an MJCF model."""

  def __init__(self, kind, mjcf_element, update_interval=1,
               buffer_size=None, delay=None,
               aggregator=None, corruptor=None):
    """Initializes this observable.

    Args:
      kind: The name of an attribute of a bound `mjcf.Physics` instance. See the
        docstring for `mjcf.Physics.bind()` for examples showing this syntax.
      mjcf_element: An `mjcf.Element`, or iterable of `mjcf.Element`.
      update_interval: (optional) An integer, number of simulation steps between
        successive updates to the value of this observable.
      buffer_size: (optional) The maximum size of the returned buffer.
        This option is only relevant when used in conjunction with an
        `observation.Updater`. If None, `observation.DEFAULT_BUFFER_SIZE` will
        be used.
      delay: (optional) Number of additional simulation steps that must be
        taken before an observation is returned. This option is only relevant
        when used in conjunction with an`observation.Updater`. If None,
        `observation.DEFAULT_DELAY` will be used.
      aggregator: (optional) Name of an item in `AGGREGATORS` or a callable that
        performs a reduction operation over the first dimension of the buffered
        observation before it is returned. A value of `None` means that no
        aggregation will be performed and the whole buffer will be returned.
      corruptor: (optional) A callable which takes a single observation as
        an argument, modifies it, and returns it. An example use case for this
        is to add random noise to the observation. When used in a
        `BufferedWrapper`, the corruptor is applied to the observation before
        it is added to the buffer. In particular, this means that the aggregator
        operates on corrupted observations.

    Raises:
      ValueError: if `mjcf_element` is not an `mjcf.Element`.
    """
    _check_mjcf_element_iterable(mjcf_element)
    self._kind = kind
    self._mjcf_element = mjcf_element
    super(MJCFFeature, self).__init__(
        update_interval, buffer_size, delay, aggregator, corruptor)

  def _callable(self, physics):
    return lambda: getattr(physics.bind(self._mjcf_element), self._kind)


class MJCFCamera(base.Observable):
  """An observable corresponding to a camera in an MJCF model."""

  def __init__(self,
               mjcf_element,
               height=240,
               width=320,
               update_interval=1,
               buffer_size=None,
               delay=None,
               aggregator=None,
               corruptor=None,
               depth=False,
               scene_option=None):
    """Initializes this observable.

    Args:
      mjcf_element: A <camera> `mjcf.Element`.
      height: (optional) An integer, the height of the rendered image.
      width: (optional) An integer, the width of the rendered image.
      update_interval: (optional) An integer, number of simulation steps between
        successive updates to the value of this observable.
      buffer_size: (optional) The maximum size of the returned buffer.
        This option is only relevant when used in conjunction with an
        `observation.Updater`. If None, `observation.DEFAULT_BUFFER_SIZE` will
        be used.
      delay: (optional) Number of additional simulation steps that must be
        taken before an observation is returned. This option is only relevant
        when used in conjunction with an`observation.Updater`. If None,
        `observation.DEFAULT_DELAY` will be used.
      aggregator: (optional) Name of an item in `AGGREGATORS` or a callable that
        performs a reduction operation over the first dimension of the buffered
        observation before it is returned. A value of `None` means that no
        aggregation will be performed and the whole buffer will be returned.
      corruptor: (optional) A callable which takes a single observation as
        an argument, modifies it, and returns it. An example use case for this
        is to add random noise to the observation. When used in a
        `BufferedWrapper`, the corruptor is applied to the observation before
        it is added to the buffer. In particular, this means that the aggregator
        operates on corrupted observations.
      depth: (optional) A boolean. If `True`, renders a depth image (1-channel)
        instead of RGB (3-channel).
      scene_option: An optional `wrapper.MjvOption` instance that can be used to
        render the scene with custom visualization options. If None then the
        default options will be used.

    Raises:
      ValueError: if `mjcf_element` is not a <camera> element.
    """
    _check_mjcf_element(mjcf_element)
    if mjcf_element.tag != 'camera':
      raise ValueError(
          'expected a <camera> element: got {}'.format(mjcf_element))
    self._mjcf_element = mjcf_element
    self._height = height
    self._width = width

    self._n_channels = 1 if depth else 3
    self._dtype = np.float32 if depth else np.uint8
    self._depth = depth
    self._scene_option = scene_option
    super(MJCFCamera, self).__init__(
        update_interval, buffer_size, delay, aggregator, corruptor)

  @property
  def height(self):
    return self._height

  @height.setter
  def height(self, value):
    self._height = value

  @property
  def width(self):
    return self._width

  @width.setter
  def width(self, value):
    self._width = value

  @property
  def array_spec(self):
    return specs.ArraySpec(
        shape=(self._height, self._width, self._n_channels), dtype=self._dtype)

  def _callable(self, physics):

    def get_observation():
      pixels = physics.render(
          height=self._height,
          width=self._width,
          camera_id=self._mjcf_element.full_identifier,
          depth=self._depth,
          scene_option=self._scene_option)
      return np.atleast_3d(pixels)

    return get_observation
