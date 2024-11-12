import torch

def fillMissingValueWithMean_X(sample:torch.Tensor) -> torch.Tensor:

    nSamples = sample.shape[0]
    for i in range(nSamples):
        numOfNanValues = torch.isnan(sample[i]).sum().item()
        if numOfNanValues > 0:
            meanValue = torch.nanmean(sample[i])
            sample[i] = torch.nan_to_num(sample[i], nan=meanValue)
    return sample


def fillMissingValueWithMedian_Y(sample:torch.Tensor) -> torch.Tensor:
    medianValue = torch.nanmedian(sample)
    sample = torch.nan_to_num(sample, nan=medianValue)
    return sample



def ScaledX(sample:torch.Tensor) -> torch.Tensor:
    dimensional_means = sample.mean(dim=(0,2), keepdim=True)
    dimensional_std = sample.std(dim=(0,2), keepdim=True)

    scaled_sample = (sample - dimensional_means) / dimensional_std
    return scaled_sample


def preProcessData(data: torch.Tensor) -> torch.Tensor:
    data = fillMissingValueWithMean_X(data)
    data = ScaledX(data)
    return data.permute(0, 2, 1)


def preProcessLabels(data:torch.Tensor) -> torch.Tensor:
    data = fillMissingValueWithMedian_Y(data)
    data = data.to(torch.int64)
    # data = torch.nn.functional.one_hot(data, len(data.unique()))
    return data