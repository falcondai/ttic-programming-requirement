#!/bin/bash

wrsync $@ --exclude=tf-log*/ --exclude=checkpoints/ --exclude=guided/
