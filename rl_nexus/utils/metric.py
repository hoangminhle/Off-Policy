import numpy as np

AGG_INIT_VALUES = {
    np.mean : 0,
    np.max  : np.NINF,
    np.min  : np.Inf
}

class Metric:
    '''
        Aggregates a scalar value such as reward or loss.

        @param short_name: Name used for column headings, like 'rew'.
        @param long_name: Full name of the metric like 'reward per step'.
        @param formatting_string: String indicating how the metric should be printed, like '{:7.5f}'.
        @param higher_is_better: Boolean to indicate whether the metric should be maximized or minimized.
        @param aggregation_fn: Function applied to aggregate values of the metric like np.max/min/mean.

    '''
    def __init__(self, short_name, long_name, formatting_string, higher_is_better, aggregation_fn=np.mean):
        if aggregation_fn not in AGG_INIT_VALUES:
            raise ValueError('{} is not a supported aggregation function. Choices: np.max/min/mean.'.format(aggregation_fn))
        self.short_name        = short_name
        self.long_name         = long_name
        self.formatting_string = formatting_string
        self.higher_is_better  = higher_is_better
        self.aggregation_fn    = aggregation_fn
        self._values           = []
        self._lifetime_value   = AGG_INIT_VALUES[aggregation_fn]
        self._lifetime_count   = 0
        self.best_value        = np.NINF if self.higher_is_better else np.Inf

    def log(self, value):
        ''' Records the latest value of the metric. '''
        self._values.append(value)

    @property
    def aggregate_value(self):
        ''' Returns the aggregated metric value. '''
        if len(self._values) > 0:
            return self.aggregation_fn(self._values)
        else:
            return 0

    @property
    def lifetime_value(self):
        ''' Returns the aggregated metric value across all condense calls. '''
        if self._lifetime_count > 0:
            return self._lifetime_value / self._lifetime_count
        else:
            return 0

    @property
    def currently_optimal(self):
        ''' Returns true if the current aggregated metric value is more
            optimal than any other time in the metric's lifetime. '''
        return (self.higher_is_better and self.aggregate_value > self.best_value) or \
            (not self.higher_is_better and self.aggregate_value < self.best_value)

    def condense_values(self):
        ''' Returns and resets the aggregate value, condensing it into a lifetime_value. '''
        if len(self._values) > 0:
            v = self.aggregate_value

            # Update the lifetime values according to the aggregation function
            if self.aggregation_fn == np.mean:
                self._lifetime_value += v * len(self._values)
                self._lifetime_count += len(self._values)
            else:
                self._lifetime_value = self.aggregation_fn([v, self._lifetime_value])
                self._lifetime_count = 1

            # Update the best_value if needed
            if (self.higher_is_better and v > self.best_value) or (not self.higher_is_better and v < self.best_value):
                self.best_value = v

            self._values.clear()
            return v
        else:
            return 0

    def __repr__(self):
        return '<Metric Object Name: \"{}\" Value: {}>'.format(self.long_name, self.aggregate_value)
