import logging
import random
import re

import emoji
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import f1_score


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def obtain_logger(logger_path: str, logger_name: str):
    logger = logging.getLogger(logger_name)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    format_settings = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                        datefmt='%m/%d/%Y %H:%M:%S')
    file_handler = logging.FileHandler(logger_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(format_settings)
    logger.addHandler(file_handler)
    return logger


def single_trait_acc_f1(predictions, labels):
    acc = (predictions == labels).float().mean()
    f1 = f1_score(y_pred=predictions.cpu().numpy(), y_true=labels.cpu().numpy(), average='macro')
    return acc, f1


def MBTI_evaluation(all_predicted, all_labels, logger):
    IE_acc, IE_f1 = single_trait_acc_f1(all_predicted[:, 0], all_labels[:, 0])
    NS_acc, NS_f1 = single_trait_acc_f1(all_predicted[:, 1], all_labels[:, 1])
    FT_acc, FT_f1 = single_trait_acc_f1(all_predicted[:, 2], all_labels[:, 2])
    PJ_acc, PJ_f1 = single_trait_acc_f1(all_predicted[:, 3], all_labels[:, 3])
    avg_acc, avg_f1 = (IE_acc + NS_acc + FT_acc + PJ_acc) / 4, (IE_f1 + NS_f1 + FT_f1 + PJ_f1) / 4
    logger.info('\n-------------------------------')
    logger.info('IE acc = {:,.4f}, IE f1 = {:,.4f}'.format(IE_acc, IE_f1))
    logger.info('NS acc = {:,.4f}, NS f1 = {:,.4f}'.format(NS_acc, NS_f1))
    logger.info('FT acc = {:,.4f}, FT f1 = {:,.4f}'.format(FT_acc, FT_f1))
    logger.info('PJ acc = {:,.4f}, PJ f1 = {:,.4f}'.format(PJ_acc, PJ_f1))
    logger.info('avg acc = {:,.4f}, avg f1 = {:,.4f}'.format(avg_acc, avg_f1))
    result = {'IE_acc': IE_acc, 'IE_f1': IE_f1, 'NS_acc': NS_acc, 'NS_f1': NS_f1, 'FT_acc': FT_acc,
              'FT_f1': FT_f1, 'PJ_acc': PJ_acc, 'PJ_f1': PJ_f1}

    return avg_acc, avg_f1, result


def FFM_evaluation(all_predicted, all_labels, logger):
    all_predicted = torch.tensor(all_predicted).long()
    all_labels = torch.tensor(all_labels).long()
    EXT_acc, EXT_f1 = single_trait_acc_f1(all_predicted[:, 0], all_labels[:, 0])
    NEU_acc, NEU_f1 = single_trait_acc_f1(all_predicted[:, 1], all_labels[:, 1])
    AGR_acc, AGR_f1 = single_trait_acc_f1(all_predicted[:, 2], all_labels[:, 2])
    CON_acc, CON_f1 = single_trait_acc_f1(all_predicted[:, 3], all_labels[:, 3])
    OPN_acc, OPN_f1 = single_trait_acc_f1(all_predicted[:, 4], all_labels[:, 4])
    avg_acc, avg_f1 = ((EXT_acc + NEU_acc + AGR_acc + CON_acc + OPN_acc) / 5.,
                       (EXT_f1 + NEU_f1 + AGR_f1 + CON_f1 + OPN_f1) / 5.)
    logger.info('\n-------------------------------')
    logger.info('EXT acc = {:,.4f}, EXT f1 = {:,.4f}'.format(EXT_acc, EXT_f1))
    logger.info('NEU acc = {:,.4f}, NEU f1 = {:,.4f}'.format(NEU_acc, NEU_f1))
    logger.info('AGR acc = {:,.4f}, AGR f1 = {:,.4f}'.format(AGR_acc, AGR_f1))
    logger.info('CON acc = {:,.4f}, CON f1 = {:,.4f}'.format(CON_acc, CON_f1))
    logger.info('OPN acc = {:,.4f}, OPN f1 = {:,.4f}'.format(OPN_acc, OPN_f1))
    logger.info('avg acc = {:,.4f}, avg f1 = {:,.4f}'.format(avg_acc, avg_f1))
    result = {'EXT_acc': EXT_acc, 'EXT_f1': EXT_f1, 'NEU_acc': NEU_acc, 'NEU_f1': NEU_f1, 'AGR_acc': AGR_acc,
              'AGR_f1': AGR_f1, 'CON_acc': CON_acc, 'CON_f1': CON_f1, 'OPN_acc': OPN_acc, 'OPN_f1': OPN_f1}
    return avg_acc, avg_f1, result


def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate the concordance correlation coefficient."""
    # Remove NaNs
    idx = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    # Pearson correlation coefficient
    r = pearsonr(y_true, y_pred)[0]

    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # Covariance
    covariance = np.cov(y_true, y_pred)[0][1]

    # CCC formula
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def FFM_detail_evaluation(fold, epoch, all_logits, all_grands, logger):
    all_predicted = torch.tensor(all_logits) > 0.5
    all_labels = torch.tensor(all_grands) > 0.5

    # Convert to numpy for PCC and CCC calculations
    predicted_np = np.array(all_logits)
    labels_np = np.array(all_grands)

    # Initialize PCC and CCC dictionaries
    PCC = {}
    CCC = {}

    # Calculate accuracy and F1 for each trait
    EXT_acc, EXT_f1 = single_trait_acc_f1(all_predicted[:, 0], all_labels[:, 0])
    NEU_acc, NEU_f1 = single_trait_acc_f1(all_predicted[:, 1], all_labels[:, 1])
    AGR_acc, AGR_f1 = single_trait_acc_f1(all_predicted[:, 2], all_labels[:, 2])
    CON_acc, CON_f1 = single_trait_acc_f1(all_predicted[:, 3], all_labels[:, 3])
    OPN_acc, OPN_f1 = single_trait_acc_f1(all_predicted[:, 4], all_labels[:, 4])

    # Calculate PCC and CCC for each trait
    for i, trait in enumerate(['EXT', 'NEU', 'AGR', 'CON', 'OPN']):
        PCC[trait] = pearsonr(predicted_np[:, i], labels_np[:, i])[0]
        CCC[trait] = concordance_correlation_coefficient(predicted_np[:, i], labels_np[:, i])

    avg_acc, avg_f1 = ((EXT_acc + NEU_acc + AGR_acc + CON_acc + OPN_acc) / 5.,
                       (EXT_f1 + NEU_f1 + AGR_f1 + CON_f1 + OPN_f1) / 5.)
    avg_PCC = np.mean([PCC[t] for t in PCC])
    avg_CCC = np.mean([CCC[t] for t in CCC])

    logger.info('\n-------------------------------')
    logger.info('EXT acc = {:,.4f}, EXT f1 = {:,.4f}, EXT PCC = {:,.4f}, EXT CCC = {:,.4f}'.format(
        EXT_acc, EXT_f1, float(PCC['EXT']), CCC['EXT']))
    logger.info('NEU acc = {:,.4f}, NEU f1 = {:,.4f}, NEU PCC = {:,.4f}, NEU CCC = {:,.4f}'.format(
        NEU_acc, NEU_f1, float(PCC['NEU']), CCC['NEU']))
    logger.info('AGR acc = {:,.4f}, AGR f1 = {:,.4f}, AGR PCC = {:,.4f}, AGR CCC = {:,.4f}'.format(
        AGR_acc, AGR_f1, float(PCC['AGR']), CCC['AGR']))
    logger.info('CON acc = {:,.4f}, CON f1 = {:,.4f}, CON PCC = {:,.4f}, CON CCC = {:,.4f}'.format(
        CON_acc, CON_f1, float(PCC['CON']), CCC['CON']))
    logger.info('OPN acc = {:,.4f}, OPN f1 = {:,.4f}, OPN PCC = {:,.4f}, OPN CCC = {:,.4f}'.format(
        OPN_acc, OPN_f1, float(PCC['OPN']), CCC['OPN']))
    logger.info('avg acc = {:,.4f}, avg f1 = {:,.4f}, avg PCC = {:,.4f}, avg CCC = {:,.4f}'.format(
        avg_acc, avg_f1, avg_PCC, avg_CCC))

    result = {'fold': fold, 'epoch': epoch, 'best_ACC': False, 'best_F1': False,
              'EXT_acc': EXT_acc.item(), 'EXT_f1': EXT_f1, 'EXT_PCC': float(PCC['EXT']), 'EXT_CCC': CCC['EXT'],
              'NEU_acc': NEU_acc.item(), 'NEU_f1': NEU_f1, 'NEU_PCC': float(PCC['NEU']), 'NEU_CCC': CCC['NEU'],
              'AGR_acc': AGR_acc.item(), 'AGR_f1': AGR_f1, 'AGR_PCC': float(PCC['AGR']), 'AGR_CCC': CCC['AGR'],
              'CON_acc': CON_acc.item(), 'CON_f1': CON_f1, 'CON_PCC': float(PCC['CON']), 'CON_CCC': CCC['CON'],
              'OPN_acc': OPN_acc.item(), 'OPN_f1': OPN_f1, 'OPN_PCC': float(PCC['OPN']), 'OPN_CCC': CCC['OPN'],
              'avg_acc': avg_acc.item(), 'avg_f1': avg_f1, 'avg_PCC': avg_PCC, 'avg_CCC': avg_CCC
              }

    return avg_acc, avg_f1, result


def remove_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')
    return pattern.sub('', text)


def text_preprocessing(text: str, lower: bool = True):
    text = emoji.demojize(text)
    text = text.replace("’", "\'")
    if lower:
        text = text.lower()
    text = remove_chinese(text)
    return text


def log_namespace(logger, namespace, title="Arguments"):
    logger.info(title + ":")
    for key, value in vars(namespace).items():
        logger.info(f"  {key}: {value}")
