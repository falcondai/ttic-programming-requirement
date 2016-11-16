#!/bin/bash

wrsync $@ --exclude=dream*/ --exclude=tf-log/ --exclude=checkpoints/ --exclude=guided/
