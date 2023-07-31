from itertools import repeat
from typing import Optional, Sequence, Tuple
from mmengine.evaluator import BaseMetric
import torch
from dataclasses import dataclass
import ssl_robustness.evaluation.imagenet_c_dataloader as imagenet_c_dataloader
import numpy as np
from mmengine.registry import METRICS
from ssl_robustness.evaluation.imagenet_c_dataloader import corruptions, curruptions_subtypes

@dataclass
class MetricPage():
    prediction: Sequence[torch.Tensor]
    actual: Sequence[torch.Tensor]
    s: int
    c: str

    def dict(self):
        return dict(actual=self.actual,
                    prediction=self.prediction,
                    s=self.s, c=self.c)

    def list(self):
        return [self.actual, self.prediction, self.s, self.c]


@METRICS.register_module()
class RobustnessErrorRate(BaseMetric):
    default_prefix="err"
    def __init__(self,
                 top_1k_rate,
                 baseline=100,
                 baseline_err_rate=43.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device, prefix)
        
        self.clean_error= top_1k_rate
        self.baseline = baseline
        self.baseline_err_rate = baseline_err_rate

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        Results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.

        """
        actual, predicted, difficulty, ctype = data_samples
        # Save the results of a batch to `self.results`
        self.results.append({
            'batch_size': len(actual),
            'correct': (predicted.argmax(dim=1) == actual).sum().cpu(),
            's': difficulty,
            'c': ctype
        })

    def compute_metrics(self, results):
        """Compute Imagenet-c metric 

        Args:
            results (dict): {
                batch_size: int
                correct: int
                s: int
                c: str
            }

        Returns:
            dict: {
                clean_error,
                err_rate,
                mCE,
                RmCE,
            }
        """

        return dict(
            err_rate=self.get_total_error_rate(results),
            clean_error=self.clean_error,
            mCE=self.get_average_error_rate(results),
            RmCE=self.get_average_relative_error_rate(results, self.clean_error),
            result=self.compute_results(results)
        )

    def get_total_error_rate(self, results):
        """ Calculate Total Error rate, 
        1 - entire dataset correct / all images 

        Args:
            results (dict): result computed by metric

        Returns:
            int: Error Rate
        """
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return ((1 - total_correct / total_size) * 100).item()

    def get_average_relative_error_rate(self, results, err_rate):
        """ Average relative Error rate given

        Args:
            results (dict): Results
            err_rate (int): clean error rate

        Returns:
            _type_: 
        """
        res = []
        for key in imagenet_c_dataloader.curruptions_subtypes:
            res.append(self.get_relative_error_rate(results, err_rate, key))
        return np.average(res).item()

    def get_average_error_rate(self, results):
        """_summary_

        Args:
            results (_type_): _description_

        Returns:
            _type_: _description_
        """
        res = [self.get_diff_score(results, key) for key in imagenet_c_dataloader.curruptions_subtypes]
            

        return np.average(res).item()

    def compute_results(self, results):
        """_summary_

        Args:
            results (_type_): _description_
        
        Returns:
            dict: Result details for each key and by difficulty
        
        """
        
        return {
            key: {
                subkey : {
                    'err': self.get_diff_score(results, subkey),
                    'by_diff': self.get_detailed_score(results, subkey)
                }
                for subkey in subset
            }
            
            for key, subset in corruptions.items()
        }
    
    
    def get_detailed_score(self, results, c: str):
        """_summary_

        Args:
            results (_type_): _description_
            c (str): _description_

        Returns:
            _type_: _description_
        """
        
        # Select all of same type, summing only difficulty
        errs = []
        
        for i in range(1, 6):

            total_correct = sum(item['correct']
                                for item in results if item['c'] == c and item['s'] == i)

            total_size = sum(item['batch_size']
                             for item in results if item['c'] == c and item['s'] == i)
            err =(1 - total_correct / total_size) * 100 # baseline is multiple of 100 so will be cancled out
            errs.append((err/self.baseline).item() * 100)
            

        return errs

  
    def get_diff_score(self, results, c: str):
        """_summary_

        Args:
            results (_type_): _description_
            c (str): _description_

        Returns:
            _type_: _description_
        """
        
        # Select all of same type, summing only difficulty
        errs = []
        for i in range(1, 6):

            total_correct = sum(item['correct']
                                for item in results if item['c'] == c and item['s'] == i)

            total_size = sum(item['batch_size']
                             for item in results if item['c'] == c and item['s'] == i)
            err =(1 - total_correct / total_size) * 100
            errs.append(err)
            

        return ((sum(errs) / sum(repeat(self.baseline, 5))) * 100).item()

    def get_relative_error_rate(self, results, err_rate, c: str):
        """_summary_

        Args:
            results (_type_): _description_
            err_rate (_type_): _description_
            c (str): _description_

        Returns:
            _type_: _description_
        """
        # Select all of same type, summing only for each difficulty
        errs = []
        for i in range(1, 6):

            total_correct = sum(item['correct']
                                for item in results if item['c'] == c and item['s'] == i)

            total_size = sum(item['batch_size']
                             for item in results if item['c'] == c and item['s'] == i)

            r_err = (1 - total_correct / total_size) * 100 
            errs.append(r_err)
        return ((sum(errs) - err_rate ) / (sum(repeat(self.baseline, 5)) - self.baseline_err_rate) ) * 100
 