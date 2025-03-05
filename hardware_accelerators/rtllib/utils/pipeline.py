import pyrtl


class _NameIndexer(object):
    """Provides internal names that are based on a prefix and an index."""

    def __init__(self, internal_prefix="_"):
        self.internal_prefix = internal_prefix
        self.internal_index = 0

    def make_valid_string(self):
        """Build a valid string based on the prefix and internal index."""
        return self.internal_prefix + str(self.next_index())

    def next_index(self):
        index = self.internal_index
        self.internal_index += 1
        return index


_piperegIndexer = _NameIndexer()


def _reset_pipereg_indexers():
    global _piperegIndexer
    _piperegIndexer = _NameIndexer()


class SimplePipeline(object):
    """Pipeline builder with auto generation of pipeline registers."""

    def __init__(self, pipeline_name=None):
        self._pipeline_name = pipeline_name or self.__class__.__name__
        self._pipeline_register_map = {}
        self._current_stage_num = 0
        stage_list = [method for method in dir(self) if method.startswith("stage")]
        for stage in sorted(stage_list):
            stage_method = getattr(self, stage)
            stage_method()
            self._current_stage_num += 1

    def __getattr__(self, name):
        try:
            return self._pipeline_register_map[self._current_stage_num][name]
        except KeyError:
            raise pyrtl.PyrtlError(
                'error, no pipeline register "%s" defined for stage %d'
                % (name, self._current_stage_num)
            )

    def __setattr__(self, name, value):
        if name.startswith("_") or name.lower() in ["e_bits", "m_bits"]:
            # do not do anything tricky with variables starting with '_'
            object.__setattr__(self, name, value)
        else:
            next_stage = self._current_stage_num + 1
            pipereg_id = str(self._current_stage_num) + "to" + str(next_stage)

            # Use the pipeline name in the register name
            rname = (
                f"{self._pipeline_name}_pipereg_{pipereg_id}_{name}"
                + _piperegIndexer.make_valid_string()
            )

            new_pipereg = pyrtl.Register(bitwidth=len(value), name=rname)
            if next_stage not in self._pipeline_register_map:
                self._pipeline_register_map[next_stage] = {}
            self._pipeline_register_map[next_stage][name] = new_pipereg
            new_pipereg.next <<= value
