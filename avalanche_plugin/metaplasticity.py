import torch
from avalanche.training.plugins import SupervisedPlugin


class MetaplasticityPlugin(SupervisedPlugin):
    """
    Implementa la Metaplasticità Sinaptica (Laborieux et al., 2021)
    per BNNs operando direttamente sui Gradienti prima di optimizer.step().
    Protegge anche i logit delle classi dei task passati.
    """

    def __init__(self, m: float = 10.0, classifier_name: str = "fc"):
        super().__init__()
        self.m = m
        self.classifier_name = classifier_name
        self.seen_classes = set()

    def before_training_exp(self, strategy, **kwargs):
        """Traccia le classi viste finora per proteggerne i logit."""
        new_classes = set(strategy.experience.classes_in_this_experience)
        self.seen_classes.update(new_classes)

    def before_update(self, strategy, **kwargs):
        """
        Intercetta i gradienti DOPO loss.backward() ma PRIMA di optimizer.step().
        Modula i gradienti e maschera i logit.
        """
        curr_task = strategy.experience.current_experience
        curr_classes = set(strategy.experience.classes_in_this_experience)
        past_classes = list(self.seen_classes - curr_classes)

        for name, param in strategy.model.named_parameters():
            if param.requires_grad and param.grad is not None:

                # -------------------------------------------------------------
                # 1. PROTEZIONE TESTA DI CLASSIFICAZIONE (Zero-Grad Logit Masking)
                # -------------------------------------------------------------
                if self.classifier_name in name:
                    if curr_task > 0 and past_classes:
                        # Annulla i gradienti verso le classi passate.
                        # Evita che Adam accumuli momentum negativo verso di esse.
                        if param.dim() > 1:  # Pesi lineari della testa
                            param.grad.data[past_classes, :] = 0.0
                        elif param.dim() == 1:  # Bias della testa
                            param.grad.data[past_classes] = 0.0
                    continue  # Non applicare la metaplasticità alla testa

                # -------------------------------------------------------------
                # 2. ALGORITMO DI METAPLASTICITA' SUI FEATURE EXTRACTORS BNN
                # -------------------------------------------------------------
                if param.dim() > 1:
                    # Direzione proposta dal gradiente (ricorda che W <- W - lr * grad,
                    # quindi il "presunto" update Uw e' positivo se il grad e' positivo).
                    # Nel paper: Uw = grad. Se (Uw * sign(W^h)) > 0, smorza il gradiente.
                    grad_direction = param.grad.data
                    w_b = torch.sign(param.data)

                    # is_opposite è VERO quando il gradiente spinge W^h verso lo zero
                    # (cioè quando grad e W^h hanno lo stesso segno, ergo si sottraggono)
                    is_opposite = (grad_direction * w_b) > 0

                    # Calcola il fattore di smorzamento
                    f_meta = 1.0 - torch.tanh(self.m * param.data) ** 2

                    # Applica f_meta al gradiente
                    # Se is_opposite è falso (stiamo allontanando W^h dallo zero), f_meta = 1 (nessun filtro)
                    param.grad.data = torch.where(
                        is_opposite,
                        param.grad.data * f_meta,
                        param.grad.data
                    )