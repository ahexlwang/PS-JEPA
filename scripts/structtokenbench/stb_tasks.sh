# shellcheck shell=bash
stb_task_spec() {
  case "$1" in
    # Functional site prediction, AUROC
    bindint)     echo "interpro.yaml                    binding_label         auroc     8" ;;
    catint)      echo "interpro.yaml                    activesite_label      auroc     8" ;;
    con)         echo "interpro.yaml                    conservedsite_label   auroc     8" ;;
    rep)         echo "interpro.yaml                    repeat_label          auroc     8" ;;
    bindbio)     echo "biolip2.yaml                     binding_label         auroc     8" ;;
    catbio)      echo "biolip2.yaml                     catalytic_label       auroc     8" ;;
    bindshake)   echo "proteinshake_binding_site.yaml   binding_site          auroc     8" ;;
    ept)         echo "proteinglue_epitope_region.yaml  epitope_label         auroc     8" ;;
    # Physicochemical property prediction, Spearman rho. atlas.yaml sets
    # regression: true and num_labels: 1, so the metric is a correlation.
    flexrmsf)    echo "atlas.yaml                       rmsf_score            spearmanr 8" ;;
    flexbfactor) echo "atlas.yaml                       bfactor_score         spearmanr 8" ;;
    flexneq)     echo "atlas.yaml                       neq_score             spearmanr 8" ;;
    # Structure property prediction, macro F1 over 45 folds
    homo)        echo "remote_homology.yaml             fold_label            f1_score  64" ;;
    *) echo "" ;;
  esac
}

STB_ALL_TASKS="bindint catint con rep bindbio catbio bindshake ept flexrmsf flexbfactor flexneq homo"
STB_FUNCTIONAL_TASKS="bindint catint con rep bindbio catbio bindshake ept"
STB_PHYSICOCHEMICAL_TASKS="flexrmsf flexbfactor flexneq"
STB_STRUCTURE_TASKS="homo"

stb_cache_key() {
  case "$1" in
    flexrmsf|flexbfactor|flexneq) echo "atlas" ;;
    *) echo "" ;;
  esac
}
