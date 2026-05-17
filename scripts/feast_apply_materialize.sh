#!/bin/bash
set -e

echo "Application de la configuration Feast..."
cd feature_repo
feast apply

echo "Matérialisation incrémentale des features vers Redis..."
# On matérialise pour les 10 dernières années vu que MovieLens date un peu
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME

echo "Matérialisation terminée avec succès !"
