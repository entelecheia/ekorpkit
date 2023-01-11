#!/bin/bash
chezmoi update --apply=false
chezmoi init --apply --verbose
chezmoi apply
