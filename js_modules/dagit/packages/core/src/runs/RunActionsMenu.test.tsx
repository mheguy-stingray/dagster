import {act, render, screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import * as React from 'react';

import {RunStatus} from '../graphql/types';
import {TestProvider} from '../testing/TestProvider';

import {RunActionsMenu} from './RunActionsMenu';
import {RunTableRunFragment} from './types/RunTable.types';

describe('RunActionsMenu', () => {
  const Test: React.FC<{permissionOverrides?: any; run: RunTableRunFragment}> = ({
    permissionOverrides,
    run,
  }) => {
    return (
      <TestProvider permissionOverrides={permissionOverrides}>
        <RunActionsMenu run={run} />
      </TestProvider>
    );
  };

  const runFragment: RunTableRunFragment = {
    __typename: 'Run',
    id: 'run-foo-bar',
    runId: 'abcdef12',
    status: RunStatus.SUCCESS,
    stepKeysToExecute: null,
    canTerminate: true,
    hasDeletePermission: true,
    hasReExecutePermission: true,
    hasTerminatePermission: true,
    mode: 'default',
    rootRunId: 'abcdef12',
    parentRunId: null,
    pipelineSnapshotId: 'snapshotID',
    parentPipelineSnapshotId: 'snapshotID',
    pipelineName: 'job-bar',
    repositoryOrigin: {
      __typename: 'RepositoryOrigin',
      id: 'repo',
      repositoryName: 'my-repo',
      repositoryLocationName: 'my-origin',
    },
    solidSelection: null,
    assetSelection: null,
    tags: [],
    startTime: 123,
    endTime: 456,
    updateTime: 789,
  };

  describe('Permissions', () => {
    it('renders menu when open', async () => {
      await act(async () => {
        render(<Test run={runFragment} />);
      });

      const button = screen.queryByRole('button') as HTMLButtonElement;
      expect(button).toBeVisible();

      await act(async () => {
        userEvent.click(button);
      });

      expect(screen.queryByRole('button', {name: /view configuration/i})).toBeVisible();
      expect(screen.queryByRole('link', {name: /open in launchpad/i})).toBeVisible();
      expect(screen.queryByRole('button', {name: /re-execute/i})).toBeVisible();
      expect(screen.queryByRole('link', {name: /download debug file/i})).toBeVisible();
      expect(screen.queryByRole('button', {name: /delete/i})).toBeVisible();
    });

    it('disables re-execution if no permission', async () => {
      await act(async () => {
        render(
          <Test
            run={runFragment}
            permissionOverrides={{
              launch_pipeline_reexecution: {enabled: false, disabledReason: 'lol nope'},
            }}
          />,
        );
      });

      const button = screen.queryByRole('button') as HTMLButtonElement;
      expect(button).toBeVisible();

      await act(async () => {
        userEvent.click(button);
      });

      const reExecutionButton = screen.queryByRole('button', {
        name: /re-execute/i,
      }) as HTMLButtonElement;

      // Blueprint doesn't actually set `disabled` on the button element.
      expect(reExecutionButton.classList.contains('bp3-disabled')).toBe(true);
    });
  });
});
