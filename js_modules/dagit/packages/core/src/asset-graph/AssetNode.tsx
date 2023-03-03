import {gql} from '@apollo/client';
import {Colors, Icon, FontFamily, Box, Caption, Spinner} from '@dagster-io/ui';
import isEqual from 'lodash/isEqual';
import React from 'react';
import {Link} from 'react-router-dom';
import styled from 'styled-components/macro';

import {withMiddleTruncation} from '../app/Util';
import {humanizedLateString, isAssetLate} from '../assets/CurrentMinutesLateTag';
import {isAssetStale, StaleCausesInfoDot} from '../assets/StaleTag';
import {assetDetailsPathForKey} from '../assets/assetDetailsPathForKey';
import {AssetComputeKindTag} from '../graph/OpTags';
import {TimestampDisplay} from '../schedules/TimestampDisplay';
import {markdownToPlaintext} from '../ui/markdownToPlaintext';

import {AssetLatestRunSpinner, AssetRunLink} from './AssetRunLinking';
import {LiveDataForNode} from './Utils';
import {ASSET_NODE_NAME_MAX_LENGTH} from './layout';
import {AssetNodeFragment} from './types/AssetNode.types';

export const AssetNode: React.FC<{
  definition: AssetNodeFragment;
  liveData?: LiveDataForNode;
  selected: boolean;
}> = React.memo(({definition, selected, liveData}) => {
  const displayName = definition.assetKey.path[definition.assetKey.path.length - 1];
  const isSource = definition.isSource;

  return (
    <AssetInsetForHoverEffect>
      <AssetNodeContainer $selected={selected}>
        <AssetNodeBox $selected={selected} $isSource={isSource}>
          <Name $isSource={isSource}>
            <span style={{marginTop: 1}}>
              <Icon name={isSource ? 'source_asset' : 'asset'} />
            </span>
            <div style={{overflow: 'hidden', textOverflow: 'ellipsis'}}>
              {withMiddleTruncation(displayName, {
                maxLength: ASSET_NODE_NAME_MAX_LENGTH,
              })}
            </div>
            <div style={{flex: 1}} />
          </Name>
          {definition.description ? (
            <Description $color={Colors.Gray800}>
              {markdownToPlaintext(definition.description).split('\n')[0]}
            </Description>
          ) : (
            <Description $color={Colors.Gray400}>No description</Description>
          )}
          {isSource && !definition.isObservable ? null : (
            <AssetNodeStatusRow definition={definition} liveData={liveData} />
          )}
          <AssetComputeKindTag definition={definition} style={{right: -2, paddingTop: 7}} />
        </AssetNodeBox>
      </AssetNodeContainer>
    </AssetInsetForHoverEffect>
  );
}, isEqual);

export const AssetNodeStatusBox: React.FC<{background: string}> = ({background, children}) => (
  <Box
    padding={{horizontal: 8}}
    style={{
      borderBottomLeftRadius: 6,
      borderBottomRightRadius: 6,
      whiteSpace: 'nowrap',
      lineHeight: '12px',
      height: 24,
    }}
    flex={{justifyContent: 'space-between', alignItems: 'center', gap: 6}}
    background={background}
  >
    {children}
  </Box>
);

interface StatusRowProps {
  definition: AssetNodeFragment;
  liveData: LiveDataForNode | undefined;
}

export const AssetNodeStatusRow: React.FC<StatusRowProps> = (props) => {
  const info = buildAssetNodeStatusRow(props);
  return <AssetNodeStatusBox background={info.background}>{info.content}</AssetNodeStatusBox>;
};

function getStepKey(definition: AssetNodeFragment) {
  // Used for linking to the run with this step highlighted. We only support highlighting
  // a single step, so just use the first one.
  const firstOp = definition.opNames.length ? definition.opNames[0] : null;
  return firstOp || '';
}

export function buildAssetNodeStatusRow({
  definition,
  liveData,
}: {
  definition: AssetNodeFragment;
  liveData: LiveDataForNode | undefined;
}) {
  if (!liveData) {
    return {
      background: Colors.Gray100,
      border: Colors.Gray300,
      content: (
        <>
          <Spinner purpose="caption-text" />
          <Caption style={{flex: 1}} color={Colors.Gray800}>
            Loading...
          </Caption>
        </>
      ),
    };
  }

  const {
    lastMaterialization,
    runWhichFailedToMaterialize,
    inProgressRunIds,
    unstartedRunIds,
  } = liveData;

  const materializingRunId = inProgressRunIds[0] || unstartedRunIds[0];
  const late = isAssetLate(liveData);

  if (definition.isSource) {
    if (materializingRunId) {
      return {
        background: Colors.Gray100,
        border: Colors.Gray300,
        content: (
          <>
            <AssetLatestRunSpinner liveData={liveData} />
            <Caption style={{flex: 1}} color={Colors.Gray800}>
              Observing...
            </Caption>
            <AssetRunLink runId={materializingRunId} />
          </>
        ),
      };
    }
    if (liveData?.lastObservation) {
      return {
        background: Colors.Gray100,
        border: Colors.Gray300,
        content: (
          <>
            <Caption>Observed</Caption>
            <Caption style={{textAlign: 'right'}}>
              <AssetRunLink
                runId={liveData.lastObservation.runId}
                event={{
                  stepKey: getStepKey(definition),
                  timestamp: liveData.lastObservation.timestamp,
                }}
              >
                <TimestampDisplay
                  timestamp={Number(liveData.lastObservation.timestamp) / 1000}
                  timeFormat={{showSeconds: false, showTimezone: false}}
                />
              </AssetRunLink>
            </Caption>
          </>
        ),
      };
    }
    return {
      background: Colors.Gray100,
      border: Colors.Gray300,
      content: (
        <>
          <Caption>Never observed</Caption>
          <Caption>–</Caption>
        </>
      ),
    };
  }

  if (materializingRunId) {
    return {
      background: Colors.Blue50,
      border: Colors.Blue500,
      content: (
        <>
          <AssetLatestRunSpinner liveData={liveData} />
          <Caption style={{flex: 1}} color={Colors.Gray800}>
            Materializing...
          </Caption>
          <AssetRunLink runId={materializingRunId} />
        </>
      ),
    };
  }

  if (liveData.partitionStats) {
    const {numPartitions, numMaterialized} = liveData.partitionStats;
    const numMissing = numPartitions - numMaterialized;
    return {
      background: late ? Colors.Red50 : numMissing ? Colors.Yellow50 : Colors.Green50,
      border: late ? Colors.Red500 : numMissing ? Colors.Yellow500 : Colors.Green500,
      content: (
        <>
          <Caption color={late ? Colors.Red700 : numMissing ? Colors.Yellow700 : Colors.Green700}>
            {`${numPartitions.toLocaleString()} partitions`}
          </Caption>
          <Caption>
            <Link
              to={assetDetailsPathForKey(definition.assetKey, {view: 'partitions'})}
              target="_blank"
              rel="noreferrer"
            >
              {late
                ? humanizedLateString(liveData.freshnessInfo.currentMinutesLate)
                : `${numMissing.toLocaleString()} missing`}
            </Link>
          </Caption>
        </>
      ),
    };
  }

  const lastMaterializationLink = lastMaterialization ? (
    <Caption>
      <AssetRunLink
        runId={lastMaterialization.runId}
        event={{stepKey: getStepKey(definition), timestamp: lastMaterialization.timestamp}}
      >
        <TimestampDisplay
          timestamp={Number(lastMaterialization.timestamp) / 1000}
          timeFormat={{showSeconds: false, showTimezone: false}}
        />
      </AssetRunLink>
    </Caption>
  ) : undefined;

  if (runWhichFailedToMaterialize || late) {
    return {
      background: Colors.Red50,
      border: Colors.Red500,
      content: (
        <>
          <Caption color={Colors.Red700}>
            {runWhichFailedToMaterialize && late
              ? `Failed (Late)`
              : late
              ? humanizedLateString(liveData.freshnessInfo.currentMinutesLate)
              : 'Failed'}
          </Caption>

          {runWhichFailedToMaterialize ? (
            <Caption>
              <AssetRunLink runId={runWhichFailedToMaterialize.id}>
                <TimestampDisplay
                  timestamp={Number(runWhichFailedToMaterialize.endTime)}
                  timeFormat={{showSeconds: false, showTimezone: false}}
                />
              </AssetRunLink>
            </Caption>
          ) : (
            lastMaterializationLink
          )}
        </>
      ),
    };
  }

  if (!lastMaterialization) {
    return {
      background: Colors.Yellow50,
      border: Colors.Yellow500,
      content: <Caption color={Colors.Yellow700}>Never materialized</Caption>,
    };
  }

  if (!liveData.freshnessPolicy && isAssetStale(liveData)) {
    return {
      background: Colors.Yellow50,
      border: Colors.Yellow500,
      content: (
        <>
          <Box flex={{gap: 4, alignItems: 'center'}}>
            <Caption color={Colors.Yellow700}>Stale</Caption>
            <StaleCausesInfoDot causes={liveData.staleStatusCauses} />
          </Box>
          {lastMaterializationLink}
        </>
      ),
    };
  }
  return {
    background: Colors.Green50,
    border: Colors.Green500,
    content: (
      <>
        <Caption color={Colors.Green700}>Materialized</Caption>
        {lastMaterializationLink}
      </>
    ),
  };
}

export const AssetNodeMinimal: React.FC<{
  selected: boolean;
  liveData?: LiveDataForNode;
  definition: AssetNodeFragment;
}> = ({selected, definition, liveData}) => {
  const {isSource, assetKey} = definition;
  const {border, background} = buildAssetNodeStatusRow({definition, liveData});
  const displayName = assetKey.path[assetKey.path.length - 1];

  return (
    <AssetInsetForHoverEffect>
      <MinimalAssetNodeContainer $selected={selected}>
        <MinimalAssetNodeBox
          $selected={selected}
          $isSource={isSource}
          $background={background}
          $border={border}
        >
          <div style={{position: 'absolute', bottom: 6, left: 6}}>
            <AssetLatestRunSpinner liveData={liveData} purpose="section" />
          </div>

          <MinimalName style={{fontSize: 30}} $isSource={isSource}>
            {withMiddleTruncation(displayName, {maxLength: 14})}
          </MinimalName>
        </MinimalAssetNodeBox>
      </MinimalAssetNodeContainer>
    </AssetInsetForHoverEffect>
  );
};

export const ASSET_NODE_LIVE_FRAGMENT = gql`
  fragment AssetNodeLiveFragment on AssetNode {
    id
    opNames
    repository {
      id
    }
    assetKey {
      path
    }
    assetMaterializations(limit: 1) {
      ...AssetNodeLiveMaterialization
    }
    freshnessPolicy {
      ...AssetNodeLiveFreshnessPolicy
    }
    freshnessInfo {
      ...AssetNodeLiveFreshnessInfo
    }
    assetObservations(limit: 1) {
      ...AssetNodeLiveObservation
    }
    staleStatus
    staleStatusCauses {
      key {
        path
      }
      reason
      dependency {
        path
      }
    }
    partitionStats {
      numMaterialized
      numPartitions
    }
  }

  fragment AssetNodeLiveFreshnessPolicy on FreshnessPolicy {
    maximumLagMinutes
    cronSchedule
    cronScheduleTimezone
  }

  fragment AssetNodeLiveFreshnessInfo on AssetFreshnessInfo {
    currentMinutesLate
  }

  fragment AssetNodeLiveMaterialization on MaterializationEvent {
    timestamp
    runId
  }

  fragment AssetNodeLiveObservation on ObservationEvent {
    timestamp
    runId
  }
`;

// Note: This fragment should only contain fields that are needed for
// useAssetGraphData and the Asset DAG. Some pages of Dagit request this
// fragment for every AssetNode on the instance. Add fields with care!
//
export const ASSET_NODE_FRAGMENT = gql`
  fragment AssetNodeFragment on AssetNode {
    id
    graphName
    hasMaterializePermission
    jobNames
    opNames
    opVersion
    description
    computeKind
    isPartitioned
    isObservable
    isSource
    assetKey {
      ...AssetNodeKey
    }
  }

  fragment AssetNodeKey on AssetKey {
    path
  }
`;

const AssetInsetForHoverEffect = styled.div`
  padding: 10px 4px 2px 4px;
  height: 100%;
`;

export const AssetNodeContainer = styled.div<{$selected: boolean}>`
  user-select: none;
  cursor: default;
  padding: 4px;
`;

const AssetNodeShowOnHover = styled.span`
  display: none;
`;

export const AssetNodeBox = styled.div<{$isSource: boolean; $selected: boolean}>`
  ${(p) =>
    p.$isSource
      ? `border: 2px dashed ${p.$selected ? Colors.Gray600 : Colors.Gray300}`
      : `border: 2px solid ${p.$selected ? Colors.Blue500 : Colors.Blue200}`};

  ${(p) =>
    p.$isSource
      ? `outline: 3px solid ${p.$selected ? Colors.Gray300 : 'transparent'}`
      : `outline: 3px solid ${p.$selected ? Colors.Blue200 : 'transparent'}`};

  background: ${Colors.White};
  border-radius: 8px;
  position: relative;
  &:hover {
    box-shadow: rgba(0, 0, 0, 0.12) 0px 2px 12px 0px;
    ${AssetNodeShowOnHover} {
      display: initial;
    }
  }
`;
const Name = styled.div<{$isSource: boolean}>`
  /** Keep in sync with DISPLAY_NAME_PX_PER_CHAR */
  display: flex;
  padding: 3px 6px;
  background: ${(p) => (p.$isSource ? Colors.Gray100 : Colors.Blue50)};
  font-family: ${FontFamily.monospace};
  border-top-left-radius: 7px;
  border-top-right-radius: 7px;
  font-weight: 600;
  gap: 4px;
`;

const MinimalAssetNodeContainer = styled(AssetNodeContainer)`
  height: 100%;
`;

const MinimalAssetNodeBox = styled.div<{
  $isSource: boolean;
  $selected: boolean;
  $background: string;
  $border: string;
}>`
  background: ${(p) => p.$background};
  ${(p) =>
    p.$isSource
      ? `border: 4px dashed ${p.$selected ? Colors.Gray500 : p.$border}`
      : `border: 4px solid ${p.$selected ? Colors.Blue500 : p.$border}`};

  ${(p) =>
    p.$isSource
      ? `outline: 8px solid ${p.$selected ? Colors.Gray300 : 'transparent'}`
      : `outline: 8px solid ${p.$selected ? Colors.Blue200 : 'transparent'}`};

  border-radius: 10px;
  position: relative;
  padding: 4px;
  height: 100%;
  min-height: 46px;
  &:hover {
    box-shadow: rgba(0, 0, 0, 0.12) 0px 2px 12px 0px;
  }
`;

const MinimalName = styled(Name)`
  font-weight: 600;
  white-space: nowrap;
  position: absolute;
  background: none;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
`;

const Description = styled.div<{$color: string}>`
  padding: 6px 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: ${(p) => p.$color};
  border-top: 1px solid ${Colors.Blue50};
  background: ${Colors.White};
  font-size: 12px;
`;
